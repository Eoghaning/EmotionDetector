import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASCADE_PATH = os.path.join(BASE_DIR, "assets", "haarcascade_frontalface_default.xml")
DATA_PATH = os.path.join(BASE_DIR, "archive", "train")
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.pth")
EMOJI_DIR = os.path.join(BASE_DIR, "assets", "emojis")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")

EMOTIONS = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Neutral']

MODEL_INPUT_SIZE = (48, 48)
NORM_MEAN = [0.5]
NORM_STD = [0.5]

SMOOTHING_WINDOW = 5
CONFIDENCE_BAR_HEIGHT = 8
EMOJI_SIZE_SCALE = 1.5
