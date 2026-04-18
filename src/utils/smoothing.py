import torch
from collections import deque

class EmotionSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, probabilities):
        self.history.append(probabilities)
        avg_probs = torch.stack(list(self.history)).mean(dim=0)
        return avg_probs
