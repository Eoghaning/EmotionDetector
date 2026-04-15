"""
GPU ACCELERATED TRAINING - RTX 5070
- Uses CUDA for 50-100x speedup
- Optimized hyperparameters for safe fine-tuning
"""
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

# Ultra-conservative fine-tuning hyperparameters
BATCH_SIZE = 128  # GPU can handle larger batches
EPOCHS = 30
LEARNING_RATE = 0.00005  # Ultra-low - 1/4 of previous
NUM_CLASSES = 7
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 8

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60)
    print("GPU ACCELERATED TRAINING: 59.12% → 62%")
    print("=" * 60)

    # Moderate augmentation (not too aggressive to preserve model)
    normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    train_transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    # Load datasets
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

    # Data loaders with GPU acceleration
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=val_sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Calculate class weights
    labels = [sample[1] for sample in full_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Model initialization
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)

    # LOAD YOUR EXISTING 59.12% MODEL
    best_val_acc = 59.12
    if os.path.exists(MODEL_PATH):
        print(f"\n✓ Loading existing model at 59.12% accuracy...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("✓ Successfully loaded checkpoint!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    else:
        print(f"✗ Model not found at {MODEL_PATH}")
        sys.exit(1)

    # Optimizer with VERY aggressive weight decay for regularization
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-2)

    # Learning rate scheduler - slower decay
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-7
    )

    print(f"\n🎯 TARGET: 62.00% | Current: {best_val_acc:.2f}%")
    print(f"⚡ GPU: RTX 5070 | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"📊 Epochs: {EPOCHS} | Early Stop: {EARLY_STOPPING_PATIENCE} epochs")
    print("=" * 60 + "\n")

    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
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
        
        # Display results
        status = "🆕 NEW BEST!" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch:2d} | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}% | {status}")
        
        # Save if improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_without_improvement = 0
            print(f"           → Model saved!\n")
            
            if best_val_acc >= 62.0:
                print("\n🎉 GOAL REACHED 62%! Training complete!\n")
                break
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement % 3 == 0:
                print(f"           → No improvement for {epochs_without_improvement} epochs\n")
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹  Early stopping: No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    # Clear GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("=" * 60)
    print(f"Training complete!")
    print(f"Final best validation accuracy: {best_val_acc:.2f}%")
    if best_val_acc >= 62.0:
        print("✓ TARGET REACHED!")
    else:
        print(f"↗ Improvement: +{best_val_acc - 59.12:.2f}% from starting point")
    print("=" * 60)

if __name__ == '__main__':
    main()
