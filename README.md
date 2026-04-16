# Face2Emoji

A real-time computer vision pipeline that classifies facial expressions using a custom PyTorch CNN and overlays a transparent emoji corresponding to the detected emotion.

**MILESTONE: Final ML Model achieved 67.0% Accuracy (April 2026)**

## Core Features
- **Real-time Inference**: Processes webcam feed at 25+ FPS on CPU.
- **8-Class Emotion Mapping**:
    _______________
    |Happy    |😀|
    |Sad      |☹️|
    |Surprise |😮|
    |Angry    |😡|
    |Disgust  |😬|
    |Fear     |😨|
    |Neutral  |😐|
    |Tongue   |😛|
    ---------------

- **Visual Feedback**: Confidence intensity bar drawn below the face bounding box.
- **Session Recording**: Save video clips of the annotated feed with the `R` key.
- **Prediction Smoothing**: Temporal averaging to prevent label jitter.

## File Architecture

face2emoji/
│
├── assets/                       # Static resources
│   ├── haarcascade_frontalface_default.xml
│   └── emojis/                   # PNG images (transparent background)
│       ├── angry.png
│       ├── disgust.png
│       ├── fear.png
│       ├── happy.png
│       ├── neutral.png
│       ├── sad.png
│       ├── surprise.png
│       └── tongue.png
│
├── data/                         # Dataset handling (excluded from git)
│   └── fer2013.csv
│
├── models/                       # Saved model weights
│   └── emotion_model.pth
│
├── recordings/                   # Output directory for saved videos
│
├── src/                          # Source code
│   ├── config.py                 # Constants and paths
│   ├── dataset.py                # PyTorch Dataset class
│   ├── model.py                  # CNN architecture definition
│   ├── train.py                  # Training script
│   ├── utils/                    # Helper modules
│   │   ├── face_detection.py
│   │   ├── overlay.py
│   │   ├── smoothing.py
│   │   └── geometric.py          # Geometric Landmark Logic
│   ├── ml_main.py                # Pure AI Detector
│   ├── geo_main.py               # Pure Geometric Detector
│   ├── hybrid_main.py            # Hybrid AI+Geo Detector
│   └── train.py                  # Training script