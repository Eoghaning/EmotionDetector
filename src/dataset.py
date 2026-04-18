import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD

class FER2013Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(MODEL_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
            ])
        else:
            self.transform = transform

        self.samples = []
        self.emotion_to_idx = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'neutral': 6
        }

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

        for emotion_name, label_idx in self.emotion_to_idx.items():
            emotion_folder = os.path.join(root_dir, emotion_name)
            if os.path.isdir(emotion_folder):
                for filename in os.listdir(emotion_folder):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(emotion_folder, filename)
                        self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import time
        img_path, label = self.samples[idx]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                image = Image.open(img_path).convert('L')
                break
            except (IOError, OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                raise RuntimeError(f"Failed to load image at {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label
