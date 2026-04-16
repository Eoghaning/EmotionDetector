import cv2
import numpy as np
import os
import sys
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.geometric import GeometricEmotionDetector

def main():
    print("Normal (Geometric) System active using MediaPipe Tasks API.")
    
    geo_detector = GeometricEmotionDetector()
    
    # Initialize Face Landmarker
    base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # Stability Buffer for landmarks
    landmark_buffer = deque(maxlen=5)
    
    cap = cv2.VideoCapture(0)
    print("Webcam opened. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            current_landmarks = np.array([(l.x, l.y) for l in face_landmarks])
            
            # Smoothing: average the last 5 frames of landmarks
            landmark_buffer.append(current_landmarks)
            smoothed_landmarks = np.mean(landmark_buffer, axis=0)
            
            # Analyze geometry
            geo_data = geo_detector.analyze(smoothed_landmarks)
            emotion = geo_data["guess"]
            conf = geo_data["confidence"]
            
            # Visuals
            h, w, _ = frame.shape
            coords = np.array([(lx * w, ly * h) for lx, ly in smoothed_landmarks])
            x_min, y_min = np.min(coords, axis=0).astype(int)
            x_max, y_max = np.max(coords, axis=0).astype(int)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.putText(frame, f"GEOMETRIC: {emotion} ({conf:.0f}%)", (x_min, y_min-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Stats display
            cv2.putText(frame, f"Mouth Ratio: {geo_data['mar']:.2f}", (10, 30), 1, 1, (255, 255, 255), 1)
            cv2.putText(frame, f"Lip Curvature: {geo_data['curvature']:.4f}", (10, 50), 1, 1, (255, 255, 255), 1)

            # Draw landmarks
            for lx, ly in smoothed_landmarks:
                cv2.circle(frame, (int(lx * w), int(ly * h)), 1, (0, 255, 0), -1)

        cv2.imshow('Geometric Only Detection', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
