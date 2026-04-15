import cv2
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CASCADE_PATH, MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD
from src.model import EmotionCNN

def main():
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return
    
    print(f"✓ Loading model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✓ Model loaded (62.25% accuracy)\n")
    
    # Load cascade
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return
    
    print("✓ Webcam opened. Press 'q' to quit.\n")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip and convert
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract and preprocess face
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, MODEL_INPUT_SIZE)
                face_roi = face_roi.astype(np.float32) / 255.0
                face_roi = (face_roi - NORM_MEAN[0]) / NORM_STD[0]
                face_tensor = torch.FloatTensor(face_roi).unsqueeze(0).unsqueeze(0).to(device)
                
                # Predict
                outputs = model(face_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                emotion_idx = np.argmax(probs)
                confidence = probs[emotion_idx]
                emotion = EMOTIONS[emotion_idx]
                
                # Draw
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{emotion} {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw bars
                bar_y = y + h + 5
                bar_w = w // 7
                for i, (prob, em) in enumerate(zip(probs, EMOTIONS)):
                    h_bar = int(prob * 40)
                    cv2.rectangle(frame, (x + i*bar_w, bar_y + 40 - h_bar),
                                (x + i*bar_w + bar_w - 1, bar_y + 40),
                                (200, 100, 0), -1)
            
            cv2.imshow("Emotion Detector - 62.25% Accuracy", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()