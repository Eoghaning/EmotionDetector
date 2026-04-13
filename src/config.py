import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE_PATH = os.path.join(BASE_DIR, "assets", "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.pth")
EMOJI_DIR = os.path.join(BASE_DIR, "assets", "emojis")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Tongue']
SMOOTHING_WINDOW = 5
CONFIDENCE_BAR_HEIGHT = 8
EMOJI_SIZE_SCALE = 1.5