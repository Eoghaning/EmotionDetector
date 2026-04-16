import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import DATA_PATH, MODEL_PATH, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD, EMOTIONS
from src.dataset import FER2013Dataset
from src.model import EmotionResNet

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 7
TARGET_ACC = 67.00

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = EmotionResNet(num_classes=NUM_CLASSES, pretrained=False).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    train_transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        transforms.RandomErasing(p=0.1)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    full_dataset = FER2013Dataset(DATA_PATH)
    indices = list(range(len(full_dataset)))
    split = int(np.floor(0.15 * len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_loader = DataLoader(FER2013Dataset(DATA_PATH, transform=train_transform), 
                              batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(FER2013Dataset(DATA_PATH, transform=val_transform), 
                            batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

    labels = np.array([sample[1] for sample in full_dataset.samples])
    class_counts = np.bincount(labels)
    weights = 1. / (class_counts + 1e-6)
    weights[1] = 0.0 
    weights[4] *= 2.0 
    weights = weights / weights.sum() * 7
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    best_val_acc = 0.0 

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                mask = (labels != 1)
                if not mask.any(): continue
                images, labels = images[mask].to(device), labels[mask].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch} | Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            if best_val_acc >= TARGET_ACC:
                break

if __name__ == '__main__':
    main()
