import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class FER2013Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pixels = np.array([int(p) for p in self.data.iloc[idx]['pixels'].split()])
        image = pixels.reshape(48, 48).astype(np.float32) / 255.0
        label = int(self.data.iloc[idx]['emotion'])
        image = torch.tensor(image).unsqueeze(0)  
        if self.transform:
            image = self.transform(image)
        return image, label