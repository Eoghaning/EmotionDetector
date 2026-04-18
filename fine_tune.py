import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PATH, MODEL_PATH, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD
from src.dataset import FER2013Dataset
from src.model import EmotionCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=" * 60)
    print("SAFE FINE-TUNING: 59.12% → 62%")
    print("=" * 60)

    BATCH_SIZE = 64
    EPOCHS = 40
    LEARNING_RATE = 0.000035
    NUM_CLASSES = 7
    VAL_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 12

    normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    train_transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    full_dataset = FER2013Dataset(DATA_PATH)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VAL_SPLIT * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_dataset = FER2013Dataset(DATA_PATH, transform=train_transform)
    val_dataset = FER2013Dataset(DATA_PATH, transform=val_transform)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=0
    )

    labels = [sample[1] for sample in full_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)

    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)

    best_val_acc = 59.12
    if os.path.exists(MODEL_PATH):
        print(f"\nLoading existing model...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Successfully loaded!")
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    else:
        print(f"✗ Model not found!")
        sys.exit(1)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=8e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=7, T_mult=1, eta_min=5e-8
    )

    print(f"\nTARGET: 62.00% | Current: {best_val_acc:.2f}%")
    print(f"LR: {LEARNING_RATE:.2e} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print("=" * 60 + "\n")

    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step()

        status = "NEW BEST!" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch:2d} | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}% | {status}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_without_improvement = 0

            if best_val_acc >= 62.0:
                print("\nGOAL REACHED 62%!\n")
                break
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping after {EARLY_STOPPING_PATIENCE} epochs")
                break

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("=" * 60)
    print(f"Final accuracy: {best_val_acc:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    main()
