import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Block 1: Input 1 channel (Grayscale) -> 64
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 2: 64 -> 128
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 3: 128 -> 256
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, groups=128),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 4: 256 -> 512
        self.b4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, groups=256),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Identity(),      # index 0
            nn.Flatten(),       # index 1
            nn.Linear(512, 256),# index 2
            nn.ReLU(),          # index 3
            nn.BatchNorm1d(256),# index 4
            nn.Dropout(0.5),    # index 5
            nn.Linear(256, num_classes) # index 6
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.classifier(x)
        return x
