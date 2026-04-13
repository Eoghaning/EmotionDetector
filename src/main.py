import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CASCADE_PATH

def test_webcam():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    print("Press 'q' or 'Q' to quit, or close the window with X")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Face2Emoji - Test Mode', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        
        if cv2.getWindowProperty('Face2Emoji - Test Mode', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()