import torch
import sys
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import DATA_PATH, MODEL_PATH, EMOTIONS, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD
from src.dataset import FER2013Dataset
from src.model import EmotionResNet

def analyze_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(EMOTIONS)
    
    # Use EmotionResNet since we updated the training script to use ResNet-18
    model = EmotionResNet(num_classes=num_classes, pretrained=False).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print("✓ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("The model in 'models/emotion_model.pth' might be the old CNN version.")
            return
    else:
        print(f"Model file not found at {MODEL_PATH}")
        return

    # Use standard validation transforms
    transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    # Try to find test data
    test_path = DATA_PATH.replace("train", "test")
    if not os.path.exists(test_path):
        test_path = DATA_PATH # Fallback
        
    print(f"Analyzing on: {test_path}")
    dataset = FER2013Dataset(test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    all_predictions = []
    all_labels = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)
    print(f"\n" + "="*30)
    print(f" OVERALL ACCURACY: {accuracy:.2f}%")
    print("="*30)
    
    print("\nClass-wise Accuracy:")
    for i, emotion in enumerate(EMOTIONS):
        mask = (all_labels == i)
        if np.sum(mask) > 0:
            class_acc = 100 * np.sum(all_predictions[mask] == all_labels[mask]) / np.sum(mask)
            print(f"{emotion:10}: {class_acc:.2f}% ({np.sum(mask)} samples)")

if __name__ == "__main__":
    analyze_model()
