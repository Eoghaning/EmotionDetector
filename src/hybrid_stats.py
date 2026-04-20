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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD, EMOJI_DIR
from src.model import EmotionResNet
from src.utils.geometric import GeometricEmotionDetector
from src.utils.hybrid_config import STRENGTHS, SAD_PHYSICAL_OVERRIDE, SMOOTHING_WINDOW, TRUTH_TABLE

EMOJI_MAP = {
    'Angry': 'angry.png',
    'Fear': 'fear.png',
    'Happy': 'happy.png',
    'Sad': 'sad.png',
    'Surprise': 'suprise.png',
    'Neutral': 'neutral.png'
}

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
    print(f"Hybrid System active. Using Manual Dashboard Strengths.")

    model = EmotionResNet(num_classes=7, pretrained=False).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

    MODEL_MAP = [0, 2, 3, 4, 5, 6]
    geo_detector = GeometricEmotionDetector()

    emoji_dict = {}
    for emo, filename in EMOJI_MAP.items():
        path = os.path.join(EMOJI_DIR, filename)
        if os.path.exists(path):
            emoji_dict[emo] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            emoji_dict[emo] = None

    base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    ai_buffer = deque(maxlen=SMOOTHING_WINDOW)
    geo_buffer = deque(maxlen=SMOOTHING_WINDOW)
    hybrid_buffer = deque(maxlen=SMOOTHING_WINDOW)

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            landmarks = [(l.x, l.y) for l in face_landmarks]
            
            h, w, _ = frame.shape
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

            geo_data = geo_detector.analyze(landmarks)
            geo_guess = geo_data["guess"]
            geo_probs = np.zeros(6)
            if geo_guess in EMOTIONS:
                geo_probs[EMOTIONS.index(geo_guess)] = 1.0
            geo_buffer.append(geo_probs)
            smooth_geo = np.mean(geo_buffer, axis=0)

            hybrid_probs = ai_probs.copy()
            if geo_guess in STRENGTHS:
                hybrid_probs[EMOTIONS.index(geo_guess)] += STRENGTHS[geo_guess]
            
            hybrid_buffer.append(hybrid_probs)
            smooth_hybrid = np.mean(hybrid_buffer, axis=0)
            
            scores = {emo: smooth_hybrid[i] * 100 for i, emo in enumerate(EMOTIONS)}
            if scores["Surprise"] >= 65:
                final_emotion = "Surprise"
            elif scores["Happy"] >= 15:
                final_emotion = "Happy"
            elif scores["Sad"] >= 75:
                final_emotion = "Sad"
            elif scores["Fear"] >= 20:
                final_emotion = "Fear"
            elif scores["Neutral"] >= 70:
                final_emotion = "Neutral"
            elif scores["Angry"] >= 60:
                final_emotion = "Angry"
            else:
                final_emotion = "Neutral"

            color = (255, 0, 0) if final_emotion == "Sad" else (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            emoji_target_w, emoji_y_offset = overlay_emoji(frame, emoji_dict, final_emotion, x_min, y_min, face_w)
            cv2.putText(frame, final_emotion, (x_min + face_w//2 + emoji_target_w//2 + 10, emoji_y_offset + int(emoji_target_w * 0.8)), 1, 1.5, (0, 0, 0), 2)
            
            ai_score = smooth_ai[np.argmax(smooth_ai)] * 100
            geo_score = geo_data["confidence"]
            hy_score = smooth_hybrid[EMOTIONS.index(final_emotion)] * 100
            
            bw = max(8, face_w // 25)
            bh_max = max(25, face_w // 6)
            chart_spacing = bh_max + 40
            label_width = 80
            
            bx = x_max + 15
            by = y_min
            
            def draw_chart(label, label_score, probs, start_y):
                cv2.putText(frame, f"{label}: {label_score:.0f}%", (bx + 6 * (bw + 3) + 10, start_y + bh_max // 2 + 5), 1, 0.7, (255, 255, 255), 2)
                for i, (p, em) in enumerate(zip(probs, EMOTIONS)):
                    h_bar = int(max(0, min(1.0, p)) * bh_max)
                    cv2.putText(frame, em[0], (bx + i * (bw + 3) + 2, start_y + bh_max + 12), 1, 0.4, (200, 200, 200), 1)
                    cv2.rectangle(frame, (bx + i * (bw + 3), start_y + bh_max - h_bar), (bx + i * (bw + 3) + bw, start_y + bh_max), (200, 100, 0), -1)

            draw_chart("AI", ai_score, smooth_ai, by)
            draw_chart("GEO", geo_score, smooth_geo, by + chart_spacing)
            draw_chart("HYBRID", hy_score, smooth_hybrid, by + chart_spacing * 2)
            
            for landmark in face_landmarks:
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)

        cv2.imshow('Emotion Detector - Hybrid System', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        if cv2.getWindowProperty('Emotion Detector - Hybrid System', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
