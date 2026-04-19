# EmotionDetector - Hugging Face Space

A real-time facial emotion detection web app built with Gradio.

## Models

- **ML (Main/Stats)**: Pure PyTorch AI classification
- **Geo (Main/Stats)**: Pure MediaPipe geometric detection
- **Hybrid (Main/Stats)**: AI + Geometric combined
- **Final (Main/Stats)**: Strict threshold with position requirements

## Thresholds (Final)

| Emotion | Threshold |
|---------|-----------|
| Surprise | >= 65 |
| Happy | >= 30 |
| Sad | >= 60 |
| Fear | >= 20 |
| Neutral | >= 70 |
| Angry | >= 48 |

## Position Requirements (Final)

- Distance: 5.5-13%
- Tilt F/B: 5.75-8.25%
- Turn L/R: -3 to 3%

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Usage

1. Open the web app
2. Allow webcam access
3. Click a model button to switch
4. Position your face and see the detected emotion

## Controls

- Click buttons to switch between 8 model variants
- Position status shows if requirements are met