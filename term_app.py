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

from src.config import MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD, EMOJI_DIR, CASCADE_PATH
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

THRESHOLDS = {"Happy": 30, "Sad": 60, "Angry": 48, "Surprise": 65, "Fear": 20, "Neutral": 70}
MODEL_MAP = [0, 2, 3, 4, 5, 6]

def overlay_emoji(frame, emoji_dict, emotion, x_min, y_min, face_w, face_h=0):
    if emotion not in emoji_dict or emoji_dict[emotion] is None:
        return 0, 0
    emoji = emoji_dict[emotion]
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
    return target_w, y_offset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Terminal Emotion Detector - 8 Models")
    print(f"Device: {device}")
    print("Loading model...")
    
    model = EmotionResNet(num_classes=7, pretrained=False).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded")
    
    base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
    mp_options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(mp_options)
    geo_detector = GeometricEmotionDetector()
    
    emoji_dict = {}
    for emo, filename in EMOJI_MAP.items():
        path = os.path.join(EMOJI_DIR, filename)
        if os.path.exists(path):
            emoji_dict[emo] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            emoji_dict[emo] = None
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    ai_buffer = deque(maxlen=8)
    geo_buffer = deque(maxlen=5)
    hybrid_buffer = deque(maxlen=8)
    
    mode = 7  # Final Main
    MODEL_NAMES = {1: "ML", 2: "ML", 3: "Geo", 4: "Geo", 5: "Hybrid", 6: "Hybrid", 7: "Final", 8: "Final"}
    
    cap = cv2.VideoCapture(0)
    print("\nControls:")
    print("  1-8: Switch models (1=ML, 2=ML Stats, 3=Geo, 4=Geo Stats, 5=Hybrid, 6=Hybrid Stats, 7=Final, 8=Final Stats)")
    print("  q:   Quit")
    print("\nRunning... Press q to quit\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            cv2.putText(display, "No Face", (w//2 - 40, h//2), 1, 1.5, (0, 0, 255), 2)
            cv2.imshow("Emotion Detector", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue
        
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
            face_tensor = transform(face_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                all_probs = torch.softmax(model(face_tensor), dim=1)[0].detach().cpu().numpy()
                ai_probs = all_probs[MODEL_MAP]
        else:
            ai_probs = np.zeros(6)
        
        ai_buffer.append(ai_probs)
        smooth_ai = np.mean(ai_buffer, axis=0)
        ai_emotion = EMOTIONS[np.argmax(smooth_ai)]
        
        geo_data = geo_detector.analyze(landmarks)
        geo_guess = geo_data["guess"] if geo_data else "Neutral"
        geo_probs = np.zeros(6)
        if geo_guess in EMOTIONS:
            geo_probs[EMOTIONS.index(geo_guess)] = 1.0
        
        hybrid_probs = ai_probs.copy()
        if geo_guess in STRENGTHS:
            hybrid_probs[EMOTIONS.index(geo_guess)] += STRENGTHS[geo_guess]
        hybrid_buffer.append(hybrid_probs)
        smooth_hybrid = np.mean(hybrid_buffer, axis=0)
        
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
        
        final_emo = "Neutral"
        final_score = 0
        if scores["Surprise"] >= 65:
            final_emo = "Surprise"
            final_score = scores["Surprise"]
        elif scores["Happy"] >= 30:
            final_emo = "Happy"
            final_score = scores["Happy"]
        elif scores["Sad"] >= 60:
            final_emo = "Sad"
            final_score = scores["Sad"]
        elif scores["Fear"] >= 20:
            final_emo = "Fear"
            final_score = scores["Fear"]
        elif scores["Neutral"] >= 70:
            final_emo = "Neutral"
            final_score = scores["Neutral"]
        elif scores["Angry"] >= 48:
            final_emo = "Angry"
            final_score = scores["Angry"]
        
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
        
        if mode == 1 or mode == 2:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            overlay_emoji(display, emoji_dict, ai_emotion, x_min, y_min, face_w)
            cv2.putText(display, f"[{MODEL_NAMES[mode]}] {ai_emotion}", (x_min, y_min - 10), 1, 1, (0, 255, 0), 2)
            if mode == 2:
                bar_y = y_max + 10
                bar_w = max(20, face_w // 8)
                for i, (prob, em) in enumerate(zip(smooth_ai, EMOTIONS)):
                    h_bar = int(prob * 40)
                    cv2.putText(display, em[0], (x_min + i*bar_w, bar_y + 55), 1, 0.4, (255, 255, 255), 1)
                    cv2.rectangle(display, (x_min + i*bar_w, bar_y + 40 - h_bar), (x_min + i*bar_w + bar_w - 2, bar_y + 40), (200, 100, 0), -1)
        
        elif mode == 3 or mode == 4:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            overlay_emoji(display, emoji_dict, geo_guess, x_min, y_min, face_w)
            cv2.putText(display, f"[{MODEL_NAMES[mode]}] {geo_guess}", (x_min, y_min - 10), 1, 1, (0, 255, 255), 2)
            if mode == 4 and geo_data:
                cv2.putText(display, f"MAR: {geo_data['mar']:.2f}", (10, 30), 1, 0.8, (0, 0, 0), 1)
                cv2.putText(display, f"Curv: {geo_data['curvature']:.4f}", (10, 50), 1, 0.8, (0, 0, 0), 1)
                for lx, ly in landmarks:
                    cv2.circle(display, (int(lx * w), int(ly * h)), 1, (0, 255, 0), -1)
        
        elif mode == 5 or mode == 6:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            overlay_emoji(display, emoji_dict, hybrid_emotion, x_min, y_min, face_w)
            cv2.putText(display, f"[{MODEL_NAMES[mode]}] {hybrid_emotion}", (x_min, y_min - 10), 1, 1, (255, 0, 0), 2)
            if mode == 6:
                bw = max(8, face_w // 20)
                bh_max = max(20, face_w // 6)
                chart_spacing = bh_max + 30
                bx, by = x_max + 15, y_min
                cv2.putText(display, f"ML: {smooth_ai[np.argmax(smooth_ai)]*100:.0f}%", (bx, by + 15), 1, 0.6, (255, 255, 255), 1)
                cv2.putText(display, f"GEO", (bx, by + chart_spacing + 15), 1, 0.6, (255, 255, 255), 1)
                cv2.putText(display, f"HYB: {smooth_hybrid[np.argmax(smooth_hybrid)]*100:.0f}%", (bx, by + chart_spacing*2 + 15), 1, 0.6, (255, 255, 255), 1)
                for landmark in face_landmarks:
                    cv2.circle(display, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)
        
        elif mode == 7 or mode == 8:
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            overlay_emoji(display, emoji_dict, final_emo, x_min, y_min, face_w)
            cv2.putText(display, f"[{MODEL_NAMES[mode]}] {final_emo} ({final_score:.0f}%)", (x_min, y_min - 10), 1, 1, (0, 255, 255), 2)
            if mode == 8:
                cv2.putText(display, f"Dist: {face_pct:.1f}%", (10, 30), 1, 0.8, (0, 0, 0), 1)
                cv2.putText(display, f"F/B: {head_tilt:.1f}%", (10, 50), 1, 0.8, (0, 0, 0), 1)
                cv2.putText(display, f"L/R: {head_turn:.1f}%", (10, 70), 1, 0.8, (0, 0, 0), 1)
                thresholds = {"Happy": 30, "Sad": 60, "Angry": 48, "Surprise": 65, "Fear": 20, "Neutral": 70}
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
        
        cv2.imshow("Emotion Detector", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key >= ord('1') and key <= ord('8'):
            mode = key - ord('0')
            print(f"Switched to: {MODEL_NAMES[mode]} ({mode})")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()