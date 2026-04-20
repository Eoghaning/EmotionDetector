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
import threading
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


class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.title("Emotion Detector - 8 Models")
        self.geometry("900x750")
        
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
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
        
        self.ai_buffer = deque(maxlen=8)
        self.geo_buffer = deque(maxlen=5)
        self.hybrid_buffer = deque(maxlen=8)
        
        self.current_mode = 7  # Final Main default (internal mode)
        self.running = True
        self.MODEL_MAP = [0, 2, 3, 4, 5, 6]
        
        self.setup_ui()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.cap = cv2.VideoCapture(0)
        print("Webcam opened")
        
        self.update_video()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        title_label = ctk.CTkLabel(self, text="Emotion Detector", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.grid(row=0, column=0, pady=(10, 5))
        
        button_frame = ctk.CTkFrame(self)
        button_frame.grid(row=1, column=0, padx=20, pady=5, sticky="n")

        self.btn_layout = [
            (0, 2, "Final Main",   7, True),
            (0, 3, "Final Stats",  8, True),
            (1, 0, "Hybrid Main",  5, False),
            (1, 1, "Hybrid Stats", 6, False),
            (1, 2, "ML Main",      1, False),
            (1, 3, "ML Stats",     2, False),
            (1, 4, "Geo Main",     3, False),
            (1, 5, "Geo Stats",    4, False),
        ]

        self.buttons = {}

        for (row, col, label, internal_mode, is_final) in self.btn_layout:
            if is_final:
                btn = ctk.CTkButton(
                    button_frame,
                    text=label,
                    command=lambda m=internal_mode: self.set_mode(m),
                    width=120,
                    height=55,
                    font=ctk.CTkFont(size=14, weight="bold"),
                    fg_color="#1F6AA5",
                    hover_color="#2D89D0",
                    border_width=2,
                    border_color="#AAAAAA"
                )
            else:
                btn = ctk.CTkButton(
                    button_frame,
                    text=label,
                    command=lambda m=internal_mode: self.set_mode(m),
                    width=120,
                    height=55,
                    font=ctk.CTkFont(size=14),
                    fg_color="#2B2B2B",
                    hover_color="#3B3B3B",
                    border_width=2,
                    border_color="#2B2B2B"
                )
            btn.grid(row=row, column=col, padx=5, pady=5)
            self.buttons[internal_mode] = btn

        self._refresh_button_states()

        self.mode_label = ctk.CTkLabel(
            self,
            text=f"Current Mode: {self._mode_name(self.current_mode)}",
            font=ctk.CTkFont(size=16)
        )
        self.mode_label.grid(row=2, column=0, pady=(5, 10))
        
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=3, column=0, padx=20, pady=(0, 10))
        
        controls_frame = ctk.CTkFrame(self)
        controls_frame.grid(row=4, column=0, pady=(0, 10))
        
        self.capture_btn = ctk.CTkButton(
            controls_frame,
            text="📷 Photo",
            command=self.capture_frame,
            width=200,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#1F6AA5",
            hover_color="#2D89D0"
        )
        self.capture_btn.pack(padx=10, pady=10)
        
        self.quit_btn = ctk.CTkButton(
            controls_frame,
            text="Quit",
            command=self.on_closing,
            width=100,
            height=30,
            fg_color="#8B0000",
            hover_color="#B22222"
        )
        self.quit_btn.pack(padx=10, pady=(0, 10))

    def _mode_name(self, internal_mode):
        names = {
            7: "Final Main", 8: "Final Stats",
            5: "Hybrid Main", 6: "Hybrid Stats",
            1: "ML Main",    2: "ML Stats",
            3: "Geo Main",   4: "Geo Stats"
        }
        return names.get(internal_mode, str(internal_mode))

    def _refresh_button_states(self):
        final_modes = {7, 8}
        for internal_mode, btn in self.buttons.items():
            is_final = internal_mode in final_modes
            is_active = internal_mode == self.current_mode

            if is_active:
                btn.configure(
                    border_width=3,
                    border_color="#FFFFFF",
                    fg_color="#1F6AA5" if is_final else "#3A3A6A",
                    hover_color="#2D89D0" if is_final else "#4A4A8A"
                )
            elif is_final:
                btn.configure(
                    border_width=2,
                    border_color="#AAAAAA",
                    fg_color="#1F6AA5",
                    hover_color="#2D89D0"
                )
            else:
                btn.configure(
                    border_width=2,
                    border_color="#2B2B2B",
                    fg_color="#2B2B2B",
                    hover_color="#3B3B3B"
                )

    def set_mode(self, internal_mode):
        self.current_mode = internal_mode
        self.mode_label.configure(text=f"Current Mode: {self._mode_name(internal_mode)}")
        self._refresh_button_states()
        print(f"Switched to: {self._mode_name(internal_mode)}")

    def overlay_emoji(self, frame, emotion, x_min, y_min, face_w):
        if emotion not in self.emoji_dict or self.emoji_dict[emotion] is None:
            return
        emoji = self.emoji_dict[emotion]
        emoji_h, emoji_w = emoji.shape[:2]
        target_w = int(face_w * 0.8)
        target_h = int(emoji_h * (target_w / emoji_w))
        emoji_resized = cv2.resize(emoji, (target_w, target_h))
        y_offset = max(0, y_min - target_h - 5)
        x_offset = x_min + face_w//2 - target_w//2
        if y_offset + target_h > frame.shape[0]:
            y_offset = max(0, frame.shape[0] - target_h)
        if x_offset + target_w > frame.shape[1]:
            x_offset = max(0, frame.shape[1] - target_w)
        if x_offset < 0:
            x_offset = 0
        if emoji_resized.shape[2] == 4:
            for c in range(3):
                frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w, c] = \
                    emoji_resized[:,:,c] * (emoji_resized[:,:,3]/255.0) + \
                    frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w, c] * \
                    (1 - emoji_resized[:,:,3]/255.0)
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
        elif scores["Fear"] >= 10:
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
                cv2.rectangle(display, (x_min + i*bar_w, bar_y + 40 - h_bar), (x_min + i*bar_w + bar_w - 2, bar_y + 40), (200, 100, 0), -1)
                
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
                cv2.rectangle(display, (bx + i*(bw+2), by + bh_max - h_bar), (bx + i*(bw+2) + bw, by + bh_max), (100, 100, 200), -1)
            
            cv2.putText(display, f"GEO", (bx, by + chart_spacing + 15), 1, 0.6, (255, 255, 255), 1)
            for i, (p, em) in enumerate(zip(geo_probs, EMOTIONS)):
                h_bar = int(max(0, min(1.0, p)) * bh_max)
                cv2.rectangle(display, (bx + i*(bw+2), by + chart_spacing + bh_max - h_bar), (bx + i*(bw+2) + bw, by + chart_spacing + bh_max), (100, 200, 100), -1)
            
            cv2.putText(display, f"HYB: {smooth_hybrid[np.argmax(smooth_hybrid)]*100:.0f}%", (bx, by + chart_spacing*2 + 15), 1, 0.6, (255, 255, 255), 1)
            for i, (p, em) in enumerate(zip(smooth_hybrid, EMOTIONS)):
                h_bar = int(max(0, min(1.0, p)) * bh_max)
                cv2.rectangle(display, (bx + i*(bw+2), by + chart_spacing*2 + bh_max - h_bar), (bx + i*(bw+2) + bw, by + chart_spacing*2 + bh_max), (200, 100, 100), -1)
            
            for landmark in face_landmarks:
                cv2.circle(display, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)
                
        elif mode == 7:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
            self.overlay_emoji(display, final_display_emo, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {final_display_emo} ({final_display_score:.0f}%)", (x_min, y_min - 10), 1, 1.2, (0, 255, 255), 3)

        elif mode == 8:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
            self.overlay_emoji(display, final_display_emo, x_min, y_min, face_w)
            cv2.putText(display, f"[{mode_names[mode]}] {final_display_emo} ({final_display_score:.0f}%)", (x_min, y_min - 10), 1, 1.2, (0, 255, 255), 3)
            
            cv2.putText(display, f"Dist: {face_pct:.1f}%", (10, 30), 1, 0.8, (0, 0, 0), 1)
            cv2.putText(display, f"F/B: {head_tilt:.1f}%", (10, 50), 1, 0.8, (0, 0, 0), 1)
            cv2.putText(display, f"L/R: {head_turn:.1f}%", (10, 70), 1, 0.8, (0, 0, 0), 1)
            
            thresholds = {"Happy": 15, "Sad": 75, "Angry": 60, "Surprise": 65, "Fear": 10, "Neutral": 70}
            for i, emo in enumerate(EMOTIONS):
                score = scores[emo]
                target = thresholds.get(emo, 0)
                color = (0, 255, 0) if score >= target else (0, 0, 255)
                cv2.putText(display, f"{emo}: {score:.0f}/{target}", (x_max + 10, y_min + 20 + i*22), 1, 0.7, color, 1)
            
            for landmark in face_landmarks:
                cv2.circle(display, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)
            
            in_range = (5.5 <= face_pct <= 13) and (5.75 <= head_tilt <= 8.25) and (-3 <= head_turn <= 3)
            if not in_range:
                adjust_msgs = []
                if face_pct < 5.5:
                    adjust_msgs.append("CLOSE")
                elif face_pct > 13:
                    adjust_msgs.append("BACK")
                if head_tilt < 5.75:
                    adjust_msgs.append("DOWN")
                elif head_tilt > 8.25:
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

        # Default filename with timestamp, e.g. "emotion_20260419_143022.jpg"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"emotion_{timestamp}.jpg"

        # Open in ~/Pictures if it exists, otherwise ~/
        pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures")
        if not os.path.isdir(pictures_dir):
            pictures_dir = os.path.expanduser("~")

        filepath = filedialog.asksaveasfilename(
            parent=self,
            title="Save photo",
            initialdir=pictures_dir,
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[
                ("JPEG image", "*.jpg *.jpeg"),
                ("PNG image", "*.png"),
                ("All files", "*.*"),
            ],
        )

        if not filepath:
            return  # User cancelled

        cv2.imwrite(filepath, display)
        print(f"Photo saved: {filepath}")

        # Update preview to show the captured frame
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