import torch
from collections import deque

class EmotionSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def update(self, probabilities):
        """
        probabilities: Tensor of shape (1, num_classes)
        """
        self.history.append(probabilities)
        # Calculate mean of probabilities in the window
        avg_probs = torch.stack(list(self.history)).mean(dim=0)
        return avg_probs
