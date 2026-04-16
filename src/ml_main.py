import cv2
import os
import sys
import torch
import numpy as np
from torchvision import transforms
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CASCADE_PATH, MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD
from src.model import EmotionResNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"AI Detector (6-Emotion) active on: {device}")
    
    # Internal model still needs 7 classes, but we only show the 6 from EMOTIONS
    model = EmotionResNet(num_classes=7, pretrained=False).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    
    MODEL_MAP = [0, 2, 3, 4, 5, 6] # Skipping Disgust (index 1)
    
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    prob_buffer = deque(maxlen=8)
    cap = cv2.VideoCapture(0)
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_tensor = transform(face_roi).unsqueeze(0).to(device)
                
                all_probs = torch.softmax(model(face_tensor), dim=1)[0].cpu().numpy()
                ai_probs = all_probs[MODEL_MAP]
                
                prob_buffer.append(ai_probs)
                smoothed_probs = np.mean(prob_buffer, axis=0)
                emotion_idx = np.argmax(smoothed_probs)
                emotion = EMOTIONS[emotion_idx]
                
                color = (255, 0, 0) if emotion == "Sad" else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{emotion}: {smoothed_probs[emotion_idx]*100:.1f}%", (x, y-10), 1, 1.2, color, 2)
                
                # Bars
                bar_y = y + h + 10
                bar_w = w // len(EMOTIONS)
                for i, (prob, em) in enumerate(zip(smoothed_probs, EMOTIONS)):
                    h_bar = int(prob * 50)
                    cv2.putText(frame, em[0], (x + i*bar_w + 5, bar_y + 65), 1, 0.4, (255, 255, 255), 1)
                    cv2.rectangle(frame, (x + i*bar_w, bar_y + 50 - h_bar), (x + i*bar_w + bar_w - 2, bar_y + 50), (200, 100, 0), -1)
            
            cv2.imshow("AI Emotion Detector (6-Emotion)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
