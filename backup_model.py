"""
Safety backup script - creates a timestamped backup of your current model
before continuing training.
"""
import shutil
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.pth")
BACKUP_DIR = os.path.join(BASE_DIR, "models", "backups")

# Create backup directory if it doesn't exist
os.makedirs(BACKUP_DIR, exist_ok=True)

if os.path.exists(MODEL_PATH):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"emotion_model_59.12pct_{timestamp}.pth")
    shutil.copy(MODEL_PATH, backup_path)
    print(f"✓ Backup created: {backup_path}")
    print(f"  Current model (59.12%) is safely backed up!")
else:
    print("Warning: No existing model found at " + MODEL_PATH)
