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
from src.utils.hybrid_config import STRENGTHS, SAD_PHYSICAL_OVERRIDE, SMOOTHING_WINDOW

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Final Filter Model Active. Applying strict thresholds.")
    
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

            # 3. HYBRID CALCULATION
            hybrid_probs = ai_probs.copy()
            if geo_guess in STRENGTHS:
                hybrid_probs[EMOTIONS.index(geo_guess)] += STRENGTHS[geo_guess]
            
            hybrid_buffer.append(hybrid_probs)
            smooth_hybrid = np.mean(hybrid_buffer, axis=0)
            
            # 4. FINAL THRESHOLD & CONFLICT LOGIC
            scores = {emo: smooth_hybrid[i] * 100 for i, emo in enumerate(EMOTIONS)}
            met_emotions = {}

            # Base Thresholds
            if scores["Fear"] >= 11: met_emotions["Fear"] = scores["Fear"]
            if scores["Neutral"] >= 70: met_emotions["Neutral"] = scores["Neutral"]
            if scores["Surprise"] >= 95: met_emotions["Surprise"] = scores["Surprise"]
            if scores["Happy"] >= 55: met_emotions["Happy"] = scores["Happy"]
            if scores["Sad"] >= 60: met_emotions["Sad"] = scores["Sad"]

            # Dynamic Anger Penalty: -1% for every 1% Neutral is above 75%
            anger_penalty = max(0, scores["Neutral"] - 75)
            adjusted_angry = scores["Angry"] - anger_penalty
            if adjusted_angry >= 45: met_emotions["Angry"] = adjusted_angry

            # CONFLICT RESOLUTION
            final_display_emo = "Neutral"
            final_display_score = scores["Neutral"]

            if met_emotions:
                # Rule 1: Non-neutral wins over Neutral
                if len(met_emotions) > 1 and "Neutral" in met_emotions:
                    del met_emotions["Neutral"]
                
                # Rule 2: Fear vs Angry formula A vs (F * 3)
                if "Fear" in met_emotions and "Angry" in met_emotions:
                    if (met_emotions["Fear"] * 3) > met_emotions["Angry"]:
                        del met_emotions["Angry"]
                    else:
                        del met_emotions["Fear"]

                # Rule 3: Highest remaining score wins
                best_emo = max(met_emotions, key=met_emotions.get)
                final_display_emo = best_emo
                final_display_score = met_emotions[best_emo]

            # 5. VISUALS
            display_text = f"{final_display_emo} ({final_display_score:.0f}%)"
            color = (0, 255, 255) # Yellow
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"FINAL: {display_text}", (x_min, y_min-10), 1, 1.5, color, 2)
            
            # Persistent Dashboard (Black text to the right)
            thresholds = {"Neutral": 70, "Angry": 45, "Surprise": 95, "Happy": 55, "Sad": 60, "Fear": 11}
            for i, emo in enumerate(EMOTIONS):
                score = scores[emo]
                # Show adjusted score for Angry in the dashboard to be clear
                disp_score = adjusted_angry if emo == "Angry" else score
                target = thresholds.get(emo, "N/A")
                dash_text = f"{emo}: {disp_score:.0f} / {target}"
                cv2.putText(frame, dash_text, (x_max + 10, y_min + 30 + i*25), 1, 1.2, (0, 0, 0), 2)

            for landmark in face_landmarks:
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)

        cv2.imshow('Final High-Confidence Model', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
