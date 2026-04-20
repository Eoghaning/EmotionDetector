import customtkinter as ctk
import cv2
import torch
import numpy as np
import os
import sys
from torchvision import transforms
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import PIL.Image
import PIL.ImageTk
import datetime
from tkinter import filedialog

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD, EMOJI_DIR
from src.model import EmotionResNet
from src.utils.geometric import GeometricEmotionDetector
from src.utils.hybrid_config import STRENGTHS, SAD_PHYSICAL_OVERRIDE, TRUTH_TABLE

EMOJI_MAP = {
    'Angry': 'angry.png',
    'Fear': 'fear.png',
    'Happy': 'happy.png',
    'Sad': 'sad.png',
    'Surprise': 'suprise.png',
    'Neutral': 'neutral.png'
}

EMOJI_REF_ORDER = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Neutral']


class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("EmotionDetector")
        self.geometry("900x800")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing on: {self.device}")

        self.model = EmotionResNet(num_classes=7, pretrained=False).to(self.device)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        print("Model loaded")

        self.base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
        self.mp_options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            output_face_blendshapes=True,
            num_faces=1)
        self.detector = None

        self.geo_detector = GeometricEmotionDetector()

        self.emoji_dict = {}
        for emo, filename in EMOJI_MAP.items():
            path = os.path.join(EMOJI_DIR, filename)
            if os.path.exists(path):
                self.emoji_dict[emo] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            else:
                self.emoji_dict[emo] = None

        # Pre-build CTk emoji images for the reference strip shown below video
        self.emoji_ctk = {}
        for emo in EMOJI_REF_ORDER:
            img_cv = self.emoji_dict.get(emo)
            if img_cv is not None:
                if img_cv.shape[2] == 4:
                    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
                else:
                    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                pil_img = PIL.Image.fromarray(img_rgb).resize((48, 48), PIL.Image.LANCZOS)
                self.emoji_ctk[emo] = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(48, 48))
            else:
                self.emoji_ctk[emo] = None

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])

        self.ai_buffer = deque(maxlen=8)
        self.geo_buffer = deque(maxlen=5)
        self.hybrid_buffer = deque(maxlen=8)

        self.current_mode = 7  # Final Main default
        self.running = True
        self.MODEL_MAP = [0, 2, 3, 4, 5, 6]
        self._advanced_open = False

        self.setup_ui()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.cap = cv2.VideoCapture(0)
        print("Webcam opened")

        self.update_video()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            self, text="EmotionDetector",
            font=ctk.CTkFont(size=26, weight="bold")
        ).grid(row=0, column=0, pady=(12, 6))

        # Primary row: centred Final Main | Final Stats (fixed half-width, not stretched)
        primary_frame = ctk.CTkFrame(self, fg_color="transparent")
        primary_frame.grid(row=1, column=0, padx=20, pady=(0, 10))

        self.btn_final_main = ctk.CTkButton(
            primary_frame, text="Final Main",
            command=lambda: self.set_mode(7),
            width=200, height=65, font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#1F6AA5", hover_color="#2D89D0",
            border_width=2, border_color="#AAAAAA", corner_radius=10,
        )
        self.btn_final_main.grid(row=0, column=0, padx=(0, 6))

        self.btn_final_stats = ctk.CTkButton(
            primary_frame, text="Final Stats",
            command=lambda: self.set_mode(8),
            width=200, height=65, font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#1F6AA5", hover_color="#2D89D0",
            border_width=2, border_color="#AAAAAA", corner_radius=10,
        )
        self.btn_final_stats.grid(row=0, column=1, padx=(6, 0))

        # Gear / advanced toggle — same width as the two Final buttons combined
        # 200 + 12 (inner gap) + 200 = 412
        gear_wrapper = ctk.CTkFrame(self, fg_color="transparent")
        gear_wrapper.grid(row=2, column=0, pady=(0, 4))
        self.gear_btn = ctk.CTkButton(
            gear_wrapper, text="Additional Model Settings  ⚙  →",
            command=self._toggle_advanced,
            width=412, height=36, font=ctk.CTkFont(size=13),
            fg_color="#2B2B2B", hover_color="#3B3B3B",
            border_width=1, border_color="#555555", corner_radius=8,
        )
        self.gear_btn.pack()

        # Collapsible advanced panel
        self.advanced_frame = ctk.CTkFrame(self)
        self.advanced_frame.grid(row=3, column=0, pady=0)
        # Spacer to maintain space when advanced is closed
        self.adv_spacer = ctk.CTkFrame(self, height=300, fg_color="transparent")
        self.adv_spacer.grid(row=3, column=0, pady=0)
        self.adv_spacer.grid_remove()  # Start hidden, shown when advanced opens
        
        adv_layout = [
            (0, 0, "Hybrid Main",  5),
            (0, 1, "Hybrid Stats", 6),
            (0, 2, "ML Main",      1),
            (0, 3, "ML Stats",     2),
            (0, 4, "Geo Main",     3),
            (0, 5, "Geo Stats",    4),
        ]
        self.adv_buttons = {}
        for (r, c, label, imode) in adv_layout:
            btn = ctk.CTkButton(
                self.advanced_frame, text=label,
                command=lambda m=imode: self.set_mode(m),
                width=200, height=65, font=ctk.CTkFont(size=18, weight="bold"),
                fg_color="#1F6AA5", hover_color="#2D89D0",
                border_width=2, border_color="#AAAAAA", corner_radius=10,
            )
            btn.grid(row=r, column=c, padx=4, pady=6)
            self.adv_buttons[imode] = btn
        
        # Initially hide advanced frame
        self.advanced_frame.grid_remove()

        # Mode label - with extra top padding to leave gap for expanded settings
        self.mode_label = ctk.CTkLabel(
            self,
            text=f"Current Mode: {self._mode_name(self.current_mode)}",
            font=ctk.CTkFont(size=15)
        )
        self.mode_label.grid(row=4, column=0, pady=(20, 2))

        # Video feed
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=5, column=0, padx=20, pady=(2, 0))

        # Emoji reference panel — bordered card with title and labels below each emoji
        emoji_panel = ctk.CTkFrame(self, corner_radius=10, border_width=2, border_color="#444444")
        emoji_panel.grid(row=6, column=0, pady=(8, 4), padx=20)

        ctk.CTkLabel(
            emoji_panel, text="Detectable Emotions",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, columnspan=len(EMOJI_REF_ORDER), pady=(8, 4), padx=16)

        for idx, emo in enumerate(EMOJI_REF_ORDER):
            img = self.emoji_ctk.get(emo)
            cell = ctk.CTkFrame(emoji_panel, fg_color="transparent")
            cell.grid(row=1, column=idx, padx=10, pady=(0, 4))
            if img:
                ctk.CTkLabel(cell, image=img, text="").pack()
            else:
                ctk.CTkLabel(cell, text="?", font=ctk.CTkFont(size=28)).pack()
            ctk.CTkLabel(
                cell, text=emo,
                font=ctk.CTkFont(size=10),
                text_color="#AAAAAA"
            ).pack(pady=(2, 6))

        # Controls
        controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        controls_frame.grid(row=7, column=0, pady=(4, 12))

        ctk.CTkButton(
            controls_frame, text="📷 Photo",
            command=self.capture_frame,
            width=200, height=40, font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#1F6AA5", hover_color="#2D89D0",
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            controls_frame, text="Quit",
            command=self.on_closing,
            width=200, height=40, font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#8B0000", hover_color="#B22222",
        ).pack(side="left", padx=8)

        self._refresh_button_states()

    def _toggle_advanced(self):
        self._advanced_open = not self._advanced_open
        if self._advanced_open:
            self.advanced_frame.grid(row=3, column=0, padx=20, pady=(0, 4))
            self.gear_btn.configure(fg_color="#3B3B6A", border_color="#8888CC", text="Additional Model Settings  ⚙  ↓")
        else:
            self.advanced_frame.grid_forget()
            self.gear_btn.configure(fg_color="#2B2B2B", border_color="#555555", text="Additional Model Settings  ⚙  →")

    def _mode_name(self, m):
        return {
            7: "Final Main", 8: "Final Stats",
            5: "Hybrid Main", 6: "Hybrid Stats",
            1: "ML Main",    2: "ML Stats",
            3: "Geo Main",   4: "Geo Stats",
        }.get(m, str(m))

    def _refresh_button_states(self):
        for imode, btn in [(7, self.btn_final_main), (8, self.btn_final_stats)]:
            active = imode == self.current_mode
            btn.configure(
                border_width=3 if active else 2,
                border_color="#FFFFFF" if active else "#AAAAAA",
            )
        for imode, btn in self.adv_buttons.items():
            active = imode == self.current_mode
            btn.configure(
                border_width=3 if active else 2,
                border_color="#FFFFFF" if active else "#AAAAAA",
                fg_color="#1F6AA5" if not active else "#1F6AA5",
                hover_color="#2D89D0",
            )

    def set_mode(self, m):
        self.current_mode = m
        self.mode_label.configure(text=f"Current Mode: {self._mode_name(m)}")
        self._refresh_button_states()
        print(f"Switched to: {self._mode_name(m)}")

    def overlay_emoji(self, frame, emotion, x_min, y_min, face_w):
        if emotion not in self.emoji_dict or self.emoji_dict[emotion] is None:
            return
        emoji = self.emoji_dict[emotion]
        emoji_h, emoji_w = emoji.shape[:2]
        target_w = int(face_w * 0.8)
        target_h = int(emoji_h * (target_w / emoji_w))
        emoji_resized = cv2.resize(emoji, (target_w, target_h))
        y_offset = max(0, y_min - target_h - 5)
        x_offset = x_min + face_w // 2 - target_w // 2
        if y_offset + target_h > frame.shape[0]:
            y_offset = max(0, frame.shape[0] - target_h)
        if x_offset + target_w > frame.shape[1]:
            x_offset = max(0, frame.shape[1] - target_w)
        if x_offset < 0:
            x_offset = 0
        if emoji_resized.shape[2] == 4:
            for c in range(3):
                frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w, c] = (
                    emoji_resized[:, :, c] * (emoji_resized[:, :, 3] / 255.0)
                    + frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w, c]
                    * (1 - emoji_resized[:, :, 3] / 255.0)
                )
        else:
            frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = emoji_resized

    def process_frame(self, frame, mode):
        if self.detector is None:
            self.detector = vision.FaceLandmarker.create_from_options(self.mp_options)

        display = frame.copy()
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            cv2.putText(display, "No Face Detected", (w//2 - 80, h//2), 1, 2, (0, 0, 0), 2)
            return display

        face_landmarks = detection_result.face_landmarks[0]
        landmarks = [(l.x, l.y) for l in face_landmarks]

        coords = np.array([(l.x * w, l.y * h) for l in face_landmarks])
        x_min, y_min = np.min(coords, axis=0).astype(int)
        x_max, y_max = np.max(coords, axis=0).astype(int)
        padding = 20
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
        face_w = x_max - x_min

        face_crop = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
        if face_crop.size > 0:
            face_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                all_probs = torch.softmax(self.model(face_tensor), dim=1)[0].detach().cpu().numpy()
                ai_probs = all_probs[self.MODEL_MAP]
        else:
            ai_probs = np.zeros(6)

        self.ai_buffer.append(ai_probs)
        smooth_ai = np.mean(self.ai_buffer, axis=0)
        ai_emotion = EMOTIONS[np.argmax(smooth_ai)]

        geo_data = self.geo_detector.analyze(landmarks)
        geo_guess = geo_data["guess"] if geo_data else "Neutral"
        geo_probs = np.zeros(6)
        if geo_guess in EMOTIONS:
            geo_probs[EMOTIONS.index(geo_guess)] = 1.0

        hybrid_probs = ai_probs.copy()
        if geo_guess in STRENGTHS:
            hybrid_probs[EMOTIONS.index(geo_guess)] += STRENGTHS[geo_guess]
        self.hybrid_buffer.append(hybrid_probs)
        smooth_hybrid = np.mean(self.hybrid_buffer, axis=0)

        hybrid_emotion = EMOTIONS[np.argmax(hybrid_probs)]
        if TRUTH_TABLE.get("SAD_OVER_ANGRY") and ai_emotion == "Angry" and geo_guess == "Sad":
            hybrid_emotion = "Sad"
        if TRUTH_TABLE.get("SURPRISE_OVER_FEAR") and ai_emotion == "Fear" and geo_guess == "Surprise":
            hybrid_emotion = "Surprise"
        if ai_emotion == "Happy" and smooth_ai[EMOTIONS.index("Happy")] > TRUTH_TABLE.get("HAPPY_THRESHOLD", 0.4):
            hybrid_emotion = "Happy"
        if SAD_PHYSICAL_OVERRIDE and geo_guess == "Sad":
            hybrid_emotion = "Sad"

        scores = {emo: smooth_hybrid[i] * 100 for i, emo in enumerate(EMOTIONS)}

        final_display_emo = "Neutral"
        final_display_score = 0
        if scores["Surprise"] >= 65:
            final_display_emo = "Surprise"
            final_display_score = scores["Surprise"]
        elif scores["Happy"] >= 15:
            final_display_emo = "Happy"
            final_display_score = scores["Happy"]
        elif scores["Sad"] >= 75:
            final_display_emo = "Sad"
            final_display_score = scores["Sad"]
        elif scores["Fear"] >= 20:
            final_display_emo = "Fear"
            final_display_score = scores["Fear"]
        elif scores["Neutral"] >= 70:
            final_display_emo = "Neutral"
            final_display_score = scores["Neutral"]
        elif scores["Angry"] >= 60:
            final_display_emo = "Angry"
            final_display_score = scores["Angry"]

        face_area = face_w * (y_max - y_min)
        frame_area = frame.shape[0] * frame.shape[1]
        face_pct = (face_area / frame_area) * 100

        nose_tip = face_landmarks[1]
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        avg_eye_y = (left_eye.y + right_eye.y) / 2
        head_tilt = (nose_tip.y - avg_eye_y) * 100
        avg_eye_x = (left_eye.x + right_eye.x) / 2
        head_turn = (nose_tip.x - avg_eye_x) * 100

        mode_names = {1: "ML Main", 2: "ML Stats", 3: "Geo Main", 4: "Geo Stats",
                      5: "Hybrid Main", 6: "Hybrid Stats", 7: "Final Main", 8: "Final Stats"}

        if mode == 1:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            self.overlay_emoji(display, ai_emotion, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {ai_emotion}", (x_min, y_min - 10), 1, 1, (0, 255, 0), 2)

        elif mode == 2:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            self.overlay_emoji(display, ai_emotion, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {ai_emotion}", (x_min, y_min - 10), 1, 1, (0, 255, 0), 2)
            bar_y = y_max + 10
            bar_w = max(20, face_w // 8)
            for i, (prob, em) in enumerate(zip(smooth_ai, EMOTIONS)):
                h_bar = int(prob * 40)
                cv2.putText(display, em[0], (x_min + i*bar_w, bar_y + 55), 1, 0.4, (255, 255, 255), 1)
                cv2.rectangle(display, (x_min + i*bar_w, bar_y + 40 - h_bar),
                              (x_min + i*bar_w + bar_w - 2, bar_y + 40), (200, 100, 0), -1)

        elif mode == 3:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            self.overlay_emoji(display, geo_guess, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {geo_guess}", (x_min, y_min - 10), 1, 1, (0, 255, 255), 2)

        elif mode == 4:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            self.overlay_emoji(display, geo_guess, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {geo_guess}", (x_min, y_min - 10), 1, 1, (0, 255, 255), 2)
            if geo_data:
                cv2.putText(display, f"MAR: {geo_data['mar']:.2f}", (10, 30), 1, 0.8, (0, 0, 0), 1)
                cv2.putText(display, f"Curv: {geo_data['curvature']:.4f}", (10, 50), 1, 0.8, (0, 0, 0), 1)
            for lx, ly in landmarks:
                cv2.circle(display, (int(lx * w), int(ly * h)), 1, (0, 255, 0), -1)

        elif mode == 5:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            self.overlay_emoji(display, hybrid_emotion, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {hybrid_emotion}", (x_min, y_min - 10), 1, 1, (255, 0, 0), 2)

        elif mode == 6:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            self.overlay_emoji(display, hybrid_emotion, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {hybrid_emotion}", (x_min, y_min - 10), 1, 1, (255, 0, 0), 2)

            bw = max(8, face_w // 20)
            bh_max = max(20, face_w // 6)
            chart_spacing = bh_max + 30
            bx, by = x_max + 15, y_min

            cv2.putText(display, f"ML: {smooth_ai[np.argmax(smooth_ai)]*100:.0f}%", (bx, by + 15), 1, 0.6, (255, 255, 255), 1)
            for i, (p, em) in enumerate(zip(smooth_ai, EMOTIONS)):
                h_bar = int(max(0, min(1.0, p)) * bh_max)
                cv2.rectangle(display, (bx + i*(bw+2), by + bh_max - h_bar),
                              (bx + i*(bw+2) + bw, by + bh_max), (100, 100, 200), -1)

            cv2.putText(display, "GEO", (bx, by + chart_spacing + 15), 1, 0.6, (255, 255, 255), 1)
            for i, (p, em) in enumerate(zip(geo_probs, EMOTIONS)):
                h_bar = int(max(0, min(1.0, p)) * bh_max)
                cv2.rectangle(display, (bx + i*(bw+2), by + chart_spacing + bh_max - h_bar),
                              (bx + i*(bw+2) + bw, by + chart_spacing + bh_max), (100, 200, 100), -1)

            cv2.putText(display, f"HYB: {smooth_hybrid[np.argmax(smooth_hybrid)]*100:.0f}%",
                        (bx, by + chart_spacing*2 + 15), 1, 0.6, (255, 255, 255), 1)
            for i, (p, em) in enumerate(zip(smooth_hybrid, EMOTIONS)):
                h_bar = int(max(0, min(1.0, p)) * bh_max)
                cv2.rectangle(display, (bx + i*(bw+2), by + chart_spacing*2 + bh_max - h_bar),
                              (bx + i*(bw+2) + bw, by + chart_spacing*2 + bh_max), (200, 100, 100), -1)

            for landmark in face_landmarks:
                cv2.circle(display, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)

        elif mode == 7:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
            self.overlay_emoji(display, final_display_emo, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {final_display_emo} ({final_display_score:.0f}%)",
                        (x_min, y_min - 10), 1, 1.2, (0, 255, 255), 3)

        elif mode == 8:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
            self.overlay_emoji(display, final_display_emo, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {final_display_emo} ({final_display_score:.0f}%)",
                        (x_min, y_min - 10), 1, 1.2, (0, 255, 255), 3)

            cv2.putText(display, f"Dist: {face_pct:.1f}%", (10, 30), 1, 0.8, (0, 0, 0), 1)
            cv2.putText(display, f"F/B: {head_tilt:.1f}%", (10, 50), 1, 0.8, (0, 0, 0), 1)
            cv2.putText(display, f"L/R: {head_turn:.1f}%", (10, 70), 1, 0.8, (0, 0, 0), 1)

            thresholds = {"Happy": 15, "Sad": 75, "Angry": 60, "Surprise": 65, "Fear": 20, "Neutral": 70}
            for i, emo in enumerate(EMOTIONS):
                score = scores[emo]
                target = thresholds.get(emo, 0)
                color = (0, 255, 0) if score >= target else (0, 0, 255)
                cv2.putText(display, f"{emo}: {score:.0f}/{target}",
                            (x_max + 10, y_min + 20 + i*22), 1, 0.7, color, 1)

            for landmark in face_landmarks:
                cv2.circle(display, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)

            in_range = (5.5 <= face_pct <= 13) and (5.5 <= head_tilt <= 8) and (-3 <= head_turn <= 3)
            if not in_range:
                adjust_msgs = []
                if face_pct < 5.5:
                    adjust_msgs.append("CLOSE")
                elif face_pct > 13:
                    adjust_msgs.append("BACK")
                if head_tilt < 5.5:
                    adjust_msgs.append("DOWN")
                elif head_tilt > 8:
                    adjust_msgs.append("UP")
                if head_turn < -3:
                    adjust_msgs.append("LEFT")
                elif head_turn > 3:
                    adjust_msgs.append("RIGHT")
                adjust_text = " ".join(adjust_msgs)
                text_w = cv2.getTextSize(adjust_text, 1, 1.5, 2)[0][0]
                text_x = x_min + (face_w // 2) - (text_w // 2)
                cv2.putText(display, adjust_text, (text_x, y_min - 10), 1, 1.5, (0, 0, 255), 2)

        return display

    def update_video(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            display = self.process_frame(frame, self.current_mode)
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            display = cv2.resize(display, (640, 480))
            photo = PIL.Image.fromarray(display)
            photo = PIL.ImageTk.PhotoImage(photo)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        self.after(30, self.update_video)

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        display = self.process_frame(frame, self.current_mode)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"emotion_{timestamp}.jpg"
        pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures")
        if not os.path.isdir(pictures_dir):
            pictures_dir = os.path.expanduser("~")

        filepath = filedialog.asksaveasfilename(
            parent=self, title="Save photo",
            initialdir=pictures_dir, initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg *.jpeg"), ("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not filepath:
            return

        cv2.imwrite(filepath, display)
        print(f"Photo saved: {filepath}")

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        display_rgb = cv2.resize(display_rgb, (640, 480))
        photo = PIL.Image.fromarray(display_rgb)
        photo = PIL.ImageTk.PhotoImage(photo)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def on_closing(self):
        self.running = False
        self.cap.release()
        if self.detector:
            self.detector.close()
        self.destroy()


if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()