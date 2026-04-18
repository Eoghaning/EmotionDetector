import torch
import torch.nn as nn
from torchvision import models

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(EmotionResNet, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=original_conv.kernel_size,
                                     stride=original_conv.stride,
                                     padding=original_conv.padding,
                                     bias=False)

        with torch.no_grad():
            self.resnet.conv1.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

EmotionCNN = EmotionResNet
