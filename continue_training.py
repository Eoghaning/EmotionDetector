"""
SAFE TRAINING CONTINUATION SCRIPT
- Loads your existing 59.12% model
- Applies improved training techniques to push toward 62%
- Only overwrites if accuracy improves
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

# Enhanced Hyperparameters for 59% -> 62%
BATCH_SIZE = 32  # Smaller batch for better gradients
EPOCHS = 50  # More epochs with patience
LEARNING_RATE = 0.0002  # Very conservative for fine-tuning
NUM_CLASSES = 7
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)
    print("SAFE CONTINUED TRAINING: 59.12% → 62%")
    print("=" * 60)

    # More aggressive augmentation to improve generalization
    normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    train_transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomCrop(MODEL_INPUT_SIZE, pad_if_needed=True),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0)

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
            print("✓ Successfully loaded checkpoint! Starting continued training...")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            best_val_acc = 0.0
    else:
        print(f"✗ Model not found at {MODEL_PATH}")
        sys.exit(1)

    # Optimizer with weight decay for regularization
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-2)

    # Cosine annealing for smoother learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=1,
        eta_min=5e-6
    )

    print(f"\n🎯 TARGET: 62.00% | Current: {best_val_acc:.2f}%")
    print(f"📊 Batch size: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LEARNING_RATE}")
    print("=" * 60 + "\n")

    # Training loop
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
            print(f"           → Model saved to {MODEL_PATH}\n")
            
            if best_val_acc >= 62.0:
                print("🎉 GOAL REACHED 62%! Training complete!\n")
                break
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement % 5 == 0:
                print(f"           → No improvement for {epochs_without_improvement} epochs\n")
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹  Early stopping: No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    print("=" * 60)
    print(f"Training complete!")
    print(f"Final best validation accuracy: {best_val_acc:.2f}%")
    if best_val_acc >= 62.0:
        print("✓ Target reached!")
    else:
        print(f"↗ Improvement: +{best_val_acc - 59.12:.2f}% from starting point")
    print("=" * 60)

if __name__ == '__main__':
    main()
