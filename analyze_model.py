import torch
import sys
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import DATA_PATH, MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD
from src.dataset import FER2013Dataset
from src.model import EmotionResNet
from src.utils.geometric import GeometricEmotionDetector

def analyze_hybrid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Hybrid Evaluation (6-Emotions) on {device}...")

    model = EmotionResNet(num_classes=7, pretrained=False).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

    geo_detector = GeometricEmotionDetector()
    base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    MODEL_MAP = [0, 2, 3, 4, 5, 6]

    transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    test_path = DATA_PATH.replace("train", "test")
    if not os.path.exists(test_path):
        test_path = DATA_PATH

    print(f"Dataset: {test_path}")
    dataset = FER2013Dataset(test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = {"AI": [], "HYBRID": [], "GT": []}
    boosts = {"Sad": 0.85, "Angry": 0.55, "Happy": 0.05, "Fear": 0.10, "Surprise": 0.20, "Neutral": 0.10}

    print("\nProcessing images...")
    for images, labels in tqdm(loader):
        img_tensor = images.to(device)
        with torch.no_grad():
            all_probs = torch.softmax(model(img_tensor), dim=1)[0].cpu().numpy()
            ai_probs = all_probs[MODEL_MAP]

        img_np = images[0].numpy().squeeze()
        img_np = ((img_np * 0.5 + 0.5) * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = detector.detect(mp_image)
        geo_guess = "Neutral"
        if detection_result.face_landmarks:
            landmarks = [(l.x, l.y) for l in detection_result.face_landmarks[0]]
            geo_data = geo_detector.analyze(landmarks)
            geo_guess = geo_data["guess"]

        hybrid_probs = ai_probs.copy()
        if geo_guess in boosts:
            hybrid_probs[EMOTIONS.index(geo_guess)] += boosts[geo_guess]

        final_emotion = EMOTIONS[np.argmax(hybrid_probs)]
        if geo_guess == "Sad": final_emotion = "Sad"

        gt_label = labels.item()
        if gt_label == 1: continue

        mapped_gt = gt_label if gt_label == 0 else gt_label - 1
        if mapped_gt > 5: mapped_gt = 5

        results["AI"].append(np.argmax(ai_probs))
        results["HYBRID"].append(EMOTIONS.index(final_emotion))
        results["GT"].append(mapped_gt)

    ai_acc = 100 * np.sum(np.array(results["AI"]) == np.array(results["GT"])) / len(results["GT"])
    hy_acc = 100 * np.sum(np.array(results["HYBRID"]) == np.array(results["GT"])) / len(results["GT"])

    print(f"\n" + "="*40)
    print(f" PURE AI ACCURACY:     {ai_acc:.2f}%")
    print(f" HYBRID ACCURACY:      {hy_acc:.2f}%")
    print(f" HYBRID IMPROVEMENT:   {hy_acc - ai_acc:+.2f}%")
    print("="*40)

if __name__ == "__main__":
    import cv2
    analyze_hybrid()
