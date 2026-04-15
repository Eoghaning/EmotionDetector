"""
CLASSIFIER-ONLY FINE-TUNING: 59.12% → 62%
Freeze all convolutional layers, only train classifier head
MUCH SAFER approach for improving from 59%
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)
    print("CLASSIFIER-ONLY FINE-TUNING: 59.12% → 62%")
    print("(All Conv layers FROZEN - only classifier trained)")
    print("=" * 60)
    
    BATCH_SIZE = 96
    EPOCHS = 50
    LEARNING_RATE = 0.001  # Higher LR - only training classifier head
    NUM_CLASSES = 7
    VAL_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15
    
    # Minimal augmentation (preserve features)
    normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    
    train_transform = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(5),
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
    
    # Class weights
    labels = [sample[1] for sample in full_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Load model
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    
    best_val_acc = 59.12
    if os.path.exists(MODEL_PATH):
        print(f"\n✓ Loading model...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("✓ Model loaded!")
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    else:
        print(f"✗ Model not found!")
        sys.exit(1)
    
    # FREEZE ALL CONVOLUTIONAL LAYERS
    print("\n🔒 Freezing convolutional layers (b1, b2, b3, b4)...")
    for param in model.b1.parameters():
        param.requires_grad = False
    for param in model.b2.parameters():
        param.requires_grad = False
    for param in model.b3.parameters():
        param.requires_grad = False
    for param in model.b4.parameters():
        param.requires_grad = False
    
    # Only classifier is trainable
    print("✓ Classifier layers UNLOCKED for training\n")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    # Only optimize classifier parameters
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                            lr=LEARNING_RATE, weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    print(f"\n🎯 TARGET: 62.00% | Current: {best_val_acc:.2f}%")
    print(f"⚡ LR: {LEARNING_RATE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
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
        scheduler.step(val_acc)
        
        status = "🆕 NEW BEST!" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch:2d} | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}% | {status}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_without_improvement = 0
            print(f"           → Model saved!\n")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\n⏹ Early stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement")
                break
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print("=" * 60)
    print(f"Final accuracy: {best_val_acc:.2f}%")
    if best_val_acc > 59.12:
        print(f"✓ Improvement: +{best_val_acc - 59.12:.2f}%")
    else:
        print(f"Accuracy same or lower - reverting to backup")
    print("=" * 60)

if __name__ == '__main__':
    main()
