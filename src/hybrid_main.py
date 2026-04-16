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
from src.config import MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD
from src.model import EmotionResNet
from src.utils.geometric import GeometricEmotionDetector
# Import the new manual dashboard
from src.utils.hybrid_config import STRENGTHS, SAD_PHYSICAL_OVERRIDE, SMOOTHING_WINDOW, TRUTH_TABLE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hybrid System active. Using Manual Dashboard Strengths.")
    
    model = EmotionResNet(num_classes=7, pretrained=False).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    
    MODEL_MAP = [0, 2, 3, 4, 5, 6] 
    geo_detector = GeometricEmotionDetector()
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
            
            # 1. AI PREDICTION
            face_crop = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
            if face_crop.size > 0:
                face_tensor = transform(face_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    all_probs = torch.softmax(model(face_tensor), dim=1)[0].cpu().numpy()
                    ai_probs = all_probs[MODEL_MAP]
            else:
                ai_probs = np.zeros(6)
            ai_buffer.append(ai_probs)
            smooth_ai = np.mean(ai_buffer, axis=0)

            # 2. GEOMETRIC PREDICTION
            geo_data = geo_detector.analyze(landmarks)
            geo_guess = geo_data["guess"]
            geo_probs = np.zeros(6)
            if geo_guess in EMOTIONS:
                geo_probs[EMOTIONS.index(geo_guess)] = 1.0
            geo_buffer.append(geo_probs)
            smooth_geo = np.mean(geo_buffer, axis=0)

            # 3. HYBRID LOGIC (Using Dashboard Strengths)
            hybrid_probs = ai_probs.copy()
            if geo_guess in STRENGTHS:
                hybrid_probs[EMOTIONS.index(geo_guess)] += STRENGTHS[geo_guess]
            
            hybrid_buffer.append(hybrid_probs)
            smooth_hybrid = np.mean(hybrid_buffer, axis=0)
            
            final_idx = np.argmax(smooth_hybrid)
            final_emotion = EMOTIONS[final_idx]
            
            # Apply Truth Table Conflict Resolution
            ai_best = EMOTIONS[np.argmax(smooth_ai)]
            if TRUTH_TABLE.get("SAD_OVER_ANGRY") and ai_best == "Angry" and geo_guess == "Sad":
                final_emotion = "Sad"
            if TRUTH_TABLE.get("SURPRISE_OVER_FEAR") and ai_best == "Fear" and geo_guess == "Surprise":
                final_emotion = "Surprise"
            
            # Happy Threshold Check
            if ai_best == "Happy" and smooth_ai[EMOTIONS.index("Happy")] > TRUTH_TABLE.get("HAPPY_THRESHOLD", 0.4):
                final_emotion = "Happy" # AI dominates if very confident

            # Apply Manual Physical Override
            if SAD_PHYSICAL_OVERRIDE and geo_guess == "Sad":
                final_emotion = "Sad"

            # 4. VISUALS
            color = (255, 0, 0) if final_emotion == "Sad" else (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Show sureness for each branch
            ai_score = smooth_ai[np.argmax(smooth_ai)] * 100
            geo_score = geo_data["confidence"]
            hy_score = smooth_hybrid[final_idx] * 100
            
            y_offset = y_min - 10
            cv2.putText(frame, f"HYBRID: {final_emotion} ({hy_score:.0f}%)", (x_min, y_offset), 1, 1.5, color, 2)
            cv2.putText(frame, f"AI: {ai_best} ({ai_score:.0f}%)", (x_min, y_offset - 50), 1, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, f"GEO: {geo_guess} ({geo_score:.0f}%)", (x_min, y_offset - 30), 1, 1.0, (255, 255, 255), 1)
            
            # Triple Charts
            bx, by = x_max + 15, y_min
            bw, bh_max = 12, 40
            
            def draw_chart(title, probs, start_y):
                cv2.putText(frame, title, (bx, start_y - 5), 1, 0.8, (255, 255, 255), 1)
                for i, (p, em) in enumerate(zip(probs, EMOTIONS)):
                    h_bar = int(max(0, min(1.0, p)) * bh_max)
                    cv2.putText(frame, em[0], (bx + i*(bw+4), start_y + bh_max + 12), 1, 0.5, (200, 200, 200), 1)
                    cv2.rectangle(frame, (bx + i*(bw+4), start_y + bh_max - h_bar), (bx + i*(bw+4) + bw, start_y + bh_max), (200, 100, 0), -1)

            draw_chart("AI Scores", smooth_ai, by)
            draw_chart("GEO Votes", smooth_geo, by + 70)
            draw_chart("HYBRID (Final)", smooth_hybrid, by + 140)
            
            for landmark in face_landmarks:
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)

        cv2.imshow('Emotion Detector - Hybrid System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
