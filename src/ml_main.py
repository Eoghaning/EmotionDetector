import cv2
import os
import sys
import torch
import numpy as np
from torchvision import transforms
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CASCADE_PATH, MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD, EMOJI_DIR
from src.model import EmotionResNet

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
    print(f"AI Detector (6-Emotion) active on: {device}")
    
    model = EmotionResNet(num_classes=7, pretrained=False).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    
    MODEL_MAP = [0, 2, 3, 4, 5, 6]
    
    emoji_dict = {}
    for emo, filename in EMOJI_MAP.items():
        path = os.path.join(EMOJI_DIR, filename)
        if os.path.exists(path):
            emoji_dict[emo] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            emoji_dict[emo] = None
    
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    prob_buffer = deque(maxlen=8)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_tensor = transform(face_roi).unsqueeze(0).to(device)
            
            all_probs = torch.softmax(model(face_tensor), dim=1)[0].detach().cpu().numpy()
            ai_probs = all_probs[MODEL_MAP]
            
            prob_buffer.append(ai_probs)
            smoothed_probs = np.mean(prob_buffer, axis=0)
            emotion_idx = np.argmax(smoothed_probs)
            emotion = EMOTIONS[emotion_idx]
            
            color = (255, 0, 0) if emotion == "Sad" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            emoji_target_w, emoji_y_offset = overlay_emoji(frame, emoji_dict, emotion, x, y, w)
            cv2.putText(frame, emotion, (x + w//2 + emoji_target_w//2 + 10, emoji_y_offset + int(emoji_target_w * 0.8)), 1, 1.5, (0, 0, 0), 2)

        cv2.imshow("AI Emotion Detector (6-Emotion)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        if cv2.getWindowProperty("AI Emotion Detector (6-Emotion)", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()