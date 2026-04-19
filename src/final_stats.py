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
    print(f"Final Model Active - 2 versions (main/stats)")
    
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
            hybrid_probs = ai_probs.copy()
            if geo_guess in STRENGTHS:
                hybrid_probs[EMOTIONS.index(geo_guess)] += STRENGTHS[geo_guess]
            
            hybrid_buffer.append(hybrid_probs)
            smooth_hybrid = np.mean(hybrid_buffer, axis=0)
            
            scores = {emo: smooth_hybrid[i] * 100 for i, emo in enumerate(EMOTIONS)}
            if scores["Surprise"] >= 65:
                final_display_emo = "Surprise"
                final_display_score = scores["Surprise"]
            elif scores["Happy"] >= 30:
                final_display_emo = "Happy"
                final_display_score = scores["Happy"]
            elif scores["Sad"] >= 60:
                final_display_emo = "Sad"
                final_display_score = scores["Sad"]
            elif scores["Fear"] >= 20:
                final_display_emo = "Fear"
                final_display_score = scores["Fear"]
            elif scores["Neutral"] >= 70:
                final_display_emo = "Neutral"
                final_display_score = scores["Neutral"]
            elif scores["Angry"] >= 48:
                final_display_emo = "Angry"
                final_display_score = scores["Angry"]
            else:
                final_display_emo = "Neutral"
                final_display_score = 0
            
            face_area = face_w * (y_max - y_min)
            frame_area = frame.shape[0] * frame.shape[1]
            face_pct = (face_area / frame_area) * 100
            if face_pct < 5.5:
                dist_text = f"Dist: {face_pct:.1f}% (MOVE CLOSER TO CAMERA)"
            elif face_pct > 13:
                dist_text = f"Dist: {face_pct:.1f}% (MOVE FURTHER FROM CAMERA)"
            else:
                dist_text = f"Dist: {face_pct:.1f}%"
            cv2.putText(frame, dist_text, (10, 30), 1, 1, (0, 0, 0), 1)
            
            nose_tip = face_landmarks[1]
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            avg_eye_y = (left_eye.y + right_eye.y) / 2
            head_tilt = (nose_tip.y - avg_eye_y) * 100
            
            if head_tilt < 5.75:
                tilt_text = f"F/B: {head_tilt:.1f}% (TILT HEAD DOWN)"
            elif head_tilt > 8.25:
                tilt_text = f"F/B: {head_tilt:.1f}% (TILT HEAD UP)"
            else:
                tilt_text = f"F/B: {head_tilt:.1f}%"
            cv2.putText(frame, tilt_text, (10, 50), 1, 1, (0, 0, 0), 1)
            
            avg_eye_x = (left_eye.x + right_eye.x) / 2
            head_turn = (nose_tip.x - avg_eye_x) * 100
            if head_turn < -3:
                turn_text = f"L/R: {head_turn:.1f}% (TURN HEAD LEFT)"
            elif head_turn > 3:
                turn_text = f"L/R: {head_turn:.1f}% (TURN HEAD RIGHT)"
            else:
                turn_text = f"L/R: {head_turn:.1f}%"
            cv2.putText(frame, turn_text, (10, 70), 1, 1, (0, 0, 0), 1)
            
            in_range = (5.5 <= face_pct <= 13) and (5.75 <= head_tilt <= 8.25) and (-3 <= head_turn <= 3)
            
            color = (0, 255, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            if in_range:
                emoji_order = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Neutral']
                small_emoji_w = int(face_w * 0.15)
                total_width = small_emoji_w * 6
                start_x = x_min + (face_w - total_width) // 2
                for idx, emo in enumerate(emoji_order):
                    emo_x = start_x + idx * small_emoji_w
                    if emo in emoji_dict and emoji_dict[emo] is not None:
                        small_emoji = cv2.resize(emoji_dict[emo], (small_emoji_w, small_emoji_w))
                        y_emoji = y_min - small_emoji_w - 5
                        if y_emoji >= 0:
                            if small_emoji.shape[2] == 4:
                                for c in range(3):
                                    frame[y_emoji:y_emoji+small_emoji_w, emo_x:emo_x+small_emoji_w, c] = \
                                        small_emoji[:,:,c] * (small_emoji[:,:,3]/255.0) + \
                                        frame[y_emoji:y_emoji+small_emoji_w, emo_x:emo_x+small_emoji_w, c] * \
                                        (1 - small_emoji[:,:,3]/255.0)
                            else:
                                frame[y_emoji:y_emoji+small_emoji_w, emo_x:emo_x+small_emoji_w] = small_emoji
                emoji_target_w, emoji_y_offset = overlay_emoji(frame, emoji_dict, final_display_emo, x_min, y_min, face_w)
                cv2.putText(frame, final_display_emo, (x_min + face_w//2 + emoji_target_w//2 + 10, emoji_y_offset + int(emoji_target_w * 0.8)), 1, 1.5, (0, 0, 0), 2)
            else:
                adjust_msgs = []
                if face_pct < 5.5:
                    adjust_msgs.append("MOVE CLOSER")
                elif face_pct > 13:
                    adjust_msgs.append("MOVE BACK")
                if head_tilt < 5.75:
                    adjust_msgs.append("TILT HEAD DOWN")
                elif head_tilt > 8.25:
                    adjust_msgs.append("TILT HEAD UP")
                if head_turn < -3:
                    adjust_msgs.append("TURN HEAD LEFT")
                elif head_turn > 3:
                    adjust_msgs.append("TURN HEAD RIGHT")
                adjust_text = " ".join(adjust_msgs)
                text_w = cv2.getTextSize(adjust_text, 1, 2, 2)[0][0]
                text_x = x_min + (face_w // 2) - (text_w // 2)
                cv2.putText(frame, adjust_text, (text_x, y_min - 10), 1, 2, (0, 0, 0), 2)
            
            thresholds = {"Happy": 30, "Sad": 60, "Angry": 48, "Surprise": 65, "Fear": 20, "Neutral": 70}
            for i, emo in enumerate(EMOTIONS):
                score = scores[emo]
                disp_score = score
                target = thresholds.get(emo, "N/A")
                dash_text = f"{emo}: {disp_score:.0f} / {target}"
                cv2.putText(frame, dash_text, (x_max + 10, y_min + 30 + i*25), 1, 1.2, (0, 0, 0), 2)
            for landmark in face_landmarks:
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)
        if not detection_result.face_landmarks:
                no_face_text = "No Face Detected"
                text_w = cv2.getTextSize(no_face_text, 1, 2, 2)[0][0]
                h, w, _ = frame.shape
                text_x = (w // 2) - (text_w // 2)
                text_y = (h // 2)
                cv2.putText(frame, no_face_text, (text_x, text_y), 1, 2, (0, 0, 0), 2)
        
        cv2.imshow('Final Model (stats)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        if cv2.getWindowProperty('Final Model (stats)', cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()