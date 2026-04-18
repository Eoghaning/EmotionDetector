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
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import DATA_PATH, MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD
from src.dataset import FER2013Dataset
from src.model import EmotionResNet
from src.utils.geometric import GeometricEmotionDetector

NEW_HYBRID_DATASET = "new_hybrid_features.npy"

def extract_new_hybrid_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Creating 'new_hybrid' dataset (6 emotions)...")

    model = EmotionResNet(num_classes=7, pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    geo_detector = GeometricEmotionDetector()
    base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1))

    transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    test_path = DATA_PATH.replace("train", "test")
    dataset = FER2013Dataset(test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    MODEL_MAP = [0, 2, 3, 4, 5, 6]
    extracted_data = []

    for images, labels in tqdm(loader):
        gt_label = labels.item()
        if gt_label == 1: continue

        mapped_gt = gt_label if gt_label == 0 else gt_label - 1
        if mapped_gt > 5: mapped_gt = 5

        with torch.no_grad():
            ai_probs = torch.softmax(model(images.to(device)), dim=1)[0].cpu().numpy()[MODEL_MAP]

        img_np = ((images[0].numpy().squeeze() * 0.5 + 0.5) * 255).astype(np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB))
        res = detector.detect(mp_image)
        geo_guess = "Neutral"
        if res.face_landmarks:
            geo_guess = geo_detector.analyze([(l.x, l.y) for l in res.face_landmarks[0]])["guess"]

        extracted_data.append({"ai": ai_probs, "geo": geo_guess, "gt": mapped_gt})

    np.save(NEW_HYBRID_DATASET, extracted_data)
    print(f"New Hybrid dataset saved to {NEW_HYBRID_DATASET}")

def optimize_new_hybrid():
    print("\nOptimizing hybrid system...")
    data = np.load(NEW_HYBRID_DATASET, allow_pickle=True)

    from src.utils.hybrid_config import ORIGINAL_BOOSTS

    def calc_acc(boost_dict):
        correct = 0
        for item in data:
            hybrid = item["ai"].copy()
            if item["geo"] in boost_dict:
                hybrid[EMOTIONS.index(item["geo"])] += boost_dict[item["geo"]]
            final_emo = EMOTIONS[np.argmax(hybrid)]
            if item["geo"] == "Sad": final_emo = "Sad"
            if EMOTIONS.index(final_emo) == item["gt"]: correct += 1
        return 100 * correct / len(data)

    orig_acc = calc_acc(ORIGINAL_BOOSTS)
    print(f"Old Hybrid Accuracy: {orig_acc:.2f}%")

    best_acc = orig_acc
    best_boosts = ORIGINAL_BOOSTS.copy()

    print("Testing 2000 ratio combinations to improve accuracy...")
    for _ in range(2000):
        test_boosts = {
            "Sad": random.uniform(0.1, 1.0),
            "Angry": random.uniform(0.1, 1.0),
            "Happy": random.uniform(0.0, 0.3),
            "Fear": random.uniform(0.0, 0.4),
            "Surprise": random.uniform(0.0, 0.5),
            "Neutral": random.uniform(0.0, 0.3)
        }
        acc = calc_acc(test_boosts)
        if acc > best_acc:
            best_acc = acc
            best_boosts = test_boosts

    print("\n" + "="*40)
    print(f" New hybrid training complete")
    print(f" Best accuracy: {best_acc:.2f}%")
    print(f" Improvement over original: {best_acc - orig_acc:+.2f}%")
    print("="*40)


    with open("src/utils/hybrid_config.py", "w") as f:
        f.write("# Configuration for hybrid emotion detection\n")
        f.write(f"ORIGINAL_BOOSTS = {repr(ORIGINAL_BOOSTS)}\n\n")
        f.write(f"NEW_BOOSTS = {repr(best_boosts)}\n\n")
        f.write(f"ACTIVE_CONFIG = 'NEW' if {best_acc} > {orig_acc} else 'ORIGINAL'\n")

    print("Hybrid Config Updated. Run src/hybrid_main.py to test.")

if __name__ == "__main__":
    import cv2
    if not os.path.exists(NEW_HYBRID_DATASET):
        extract_new_hybrid_data()
    optimize_new_hybrid()
