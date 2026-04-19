import cv2
import torch
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
from collections import deque

EMOJI_DIR = "assets/emojis"
MODEL_PATH = "models/emotion_model.pth"
CASCADE_PATH = "assets/haarcascade_frontalface_default.xml"
LANDMARKER_PATH = "assets/face_landmarker.task"

EMOTIONS = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Neutral']
EMOJI_MAP = {
    'Angry': 'angry.png',
    'Fear': 'fear.png',
    'Happy': 'happy.png',
    'Sad': 'sad.png',
    'Surprise': 'suprise.png',
    'Neutral': 'neutral.png'
}
THRESHOLDS = {"Happy": 30, "Sad": 60, "Angry": 48, "Surprise": 65, "Fear": 20, "Neutral": 70}
SMOOTHING_WINDOW = 5
MODEL_INPUT_SIZE = (48, 48)
NORM_MEAN = [0.5]
NORM_STD = [0.5]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmotionResNet(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(256 * 6 * 6, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

STRENGTHS = {"Happy": 0.5, "Sad": 0.3, "Angry": 0.3, "Surprise": 0.6, "Fear": 0.3, "Neutral": 0.2}

model = EmotionResNet(num_classes=7).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

MODEL_MAP = [0, 2, 3, 4, 5, 6]

emoji_dict = {}
for emo, fn in EMOJI_MAP.items():
    path = os.path.join(EMOJI_DIR, fn)
    emoji_dict[emo] = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
ai_buffer = deque(maxlen=SMOOTHING_WINDOW)
hybrid_buffer = deque(maxlen=SMOOTHING_WINDOW)

def get_emotion(scores):
    order = ["Surprise", "Happy", "Sad", "Fear", "Neutral", "Angry"]
    for e in order:
        if scores.get(e, 0) >= THRESHOLDS.get(e, 0):
            return e
    return "Neutral"

def overlay_emoji(frame, emo, x, y, w):
    if emo not in emoji_dict or emoji_dict[emo] is None:
        return frame
    emoji = emoji_dict[emo]
    tw = int(w * 0.8)
    th = int(emoji.shape[0] * (tw / emoji.shape[1]))
    resized = cv2.resize(emoji, (tw, th))
    yo = max(0, y - th - 5)
    xo = x + w // 2 - tw // 2
    if resized.shape[2] == 4:
        for c in range(3):
            a = resized[:, :, 3] / 255.0
            frame[yo:yo+th, xo:xo+tw, c] = resized[:, :, c] * a + frame[yo:yo+th, xo:xo+tw, c] * (1 - a)
    return frame

def analyze_landmarks(landmarks):
    lip_up = landmarks[13].y
    lip_down = landmarks[14].y
    lip_curvature = lip_down - lip_up
    
    left_eyebrow = landmarks[107].y
    right_eyebrow = landmarks[336].y
    left_eye = landmarks[159].y
    right_eye = landmarks[386].y
    eyebrow_raise = (left_eye + right_eye) / 2 - (left_eyebrow + right_eyebrow) / 2
    
    nose = landmarks[1].y
    mouth = landmarks[13].y
    face_lower = landmarks[152].y
    mouth_pos = (mouth - nose) / (face_lower - nose) if face_lower != nose else 0.5
    
    if eyebrow_raise > 0.03:
        return "Surprise"
    elif lip_curvature > 0.03:
        return "Happy"
    elif mouth_pos > 0.6:
        return "Sad"
    elif mouth_pos < 0.35:
        return "Angry"
    else:
        return "Neutral"

current_model = ["ml_main"]

def process_frame(frame, model_name):
    frame = cv2.flip(frame, 1)
    global ai_buffer, hybrid_buffer
    
    emotion = "Neutral"
    positioned = "OK"
    
    if model_name in ["ml_main", "ml_stats"]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi = gray[y:y+h, x:x+w]
            tensor = transform(roi).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()[MODEL_MAP]
            ai_buffer.append(probs)
            smooth = np.mean(ai_buffer, axis=0)
            scores = {e: smooth[i] * 100 for i, e in enumerate(EMOTIONS)}
            emotion = get_emotion(scores)
            frame = overlay_emoji(frame, emotion, x, y, w)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    elif model_name in ["geo_main", "geo_stats"]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.face_landmarks:
            lms = [(l.x, l.y) for l in result.face_landmarks[0]]
            emotion = analyze_landmarks(lms)
            h, w, _ = frame.shape
            for lm in result.face_landmarks[0]:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
    
    elif model_name in ["hybrid_main", "hybrid_stats", "final_main", "final_stats"]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.face_landmarks:
            lms = [(l.x, l.y) for l in result.face_landmarks[0]]
            h, w, _ = frame.shape
            coords = np.array([(l.x * w, l.y * h) for l in result.face_landmarks[0]])
            xm, ym = np.min(coords, axis=0).astype(int)
            xMa, yMa = np.max(coords, axis=0).astype(int)
            pad = 20
            xm = max(0, xm - pad)
            ym = max(0, ym - pad)
            xMa = min(w, xMa + pad)
            yMa = min(h, yMa + pad)
            fw = xMa - xm
            
            face_crop = cv2.cvtColor(frame[ym:yMa, xm:xMa], cv2.COLOR_BGR2GRAY)
            if face_crop.size > 0:
                tensor = transform(face_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()[MODEL_MAP]
                ai_buffer.append(probs)
                smooth_ai = np.mean(ai_buffer, axis=0)
                
                geo_guess = analyze_landmarks(lms)
                hybrid_probs = smooth_ai.copy()
                if geo_guess in STRENGTHS:
                    hybrid_probs[EMOTIONS.index(geo_guess)] += STRENGTHS[geo_guess]
                
                hybrid_buffer.append(hybrid_probs)
                smooth_hybrid = np.mean(hybrid_buffer, axis=0)
                scores = {e: smooth_hybrid[i] * 100 for i, e in enumerate(EMOTIONS)}
                emotion = get_emotion(scores)
            
            if model_name.startswith("final"):
                face_area = fw * (yMa - ym)
                frame_area = frame.shape[0] * frame.shape[1]
                face_pct = (face_area / frame_area) * 100
                
                nose = lms[1]
                left_eye = lms[33]
                right_eye = lms[263]
                avg_eye_y = (left_eye.y + right_eye.y) / 2
                head_tilt = (nose.y - avg_eye_y) * 100
                
                avg_eye_x = (left_eye.x + right_eye.x) / 2
                head_turn = (nose.x - avg_eye_x) * 100
                
                if not (5.5 <= face_pct <= 13 and 5.75 <= head_tilt <= 8.25 and -3 <= head_turn <= 3):
                    positioned = "Not Positioned"
                    emotion = "Neutral"
            
            frame = overlay_emoji(frame, emotion, xm, ym, fw)
            cv2.rectangle(frame, (xm, ym), (xMa, yMa), (0, 255, 255), 2)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame), emotion, positioned

with gr.Blocks() as demo:
    gr.Markdown("# EmotionDetector")
    
    with gr.Row():
        webcam = gr.Image(sources=["webcam"], type="numpy", label="Webcam")
        output = gr.Image(type="pil", label="Output")
    
    with gr.Row():
        emotion_out = gr.Text(label="Emotion")
        position_out = gr.Text(label="Position")
    
    gr.Markdown("### Select Model")
    
    with gr.Row():
        btn_ml = gr.Button("ML")
        btn_geo = gr.Button("Geo")
        btn_hybrid = gr.Button("Hybrid")
        btn_final = gr.Button("Final")
    
    current = gr.State("ml_main")
    
    btn_ml.click(lambda: process_frame(webcam, "ml_main"), [webcam], [output, emotion_out, position_out])
    btn_geo.click(lambda: process_frame(webcam, "geo_main"), [webcam], [output, emotion_out, position_out])
    btn_hybrid.click(lambda: process_frame(webcam, "hybrid_main"), [webcam], [output, emotion_out, position_out])
    btn_final.click(lambda: process_frame(webcam, "final_main"), [webcam], [output, emotion_out, position_out])

demo.launch()