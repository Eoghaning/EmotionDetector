# EmotionDetector

A real-time computer vision system that classifies facial expressions and overlays emojis. Built with a goal of achieving accurate face detection through multiple approaches, ending up with 4 distinct model types each with 2 versions (main/stats).

## Goal

Create an accurate real-time facial emotion detection system that reliably identifies emotions and provides visual feedback through emojis, with different approaches for different use cases.

## 8 Model Files (4 models × 2 versions)

| Model | Description |
|-------|------------|
| **ml_main.py** | Pure PyTorch CNN - AI-based emotion classification |
| **ml_stats.py** | Same as ml_main with probability bars visualization |
| **geo_main.py** | Pure geometric landmarks - rule-based detection |
| **geo_stats.py** | Same as geo_main with landmarks visualization |
| **hybrid_main.py** | AI + Geometric combined - hybrid approach |
| **hybrid_stats.py** | Same as hybrid_main with stats display |
| **final_main.py** | Strict threshold checklist - most refined |
| **final_stats.py** | Same as final_main with full UI statistics |

## Model Details

### ML Models (ml_main.py, ml_stats.py)
- **Tool**: PyTorch CNN (EmotionResNet)
- **Detection**: Haar Cascade (OpenCV)
- **Approach**: Pure AI-based classification with temporal smoothing
- **Use case**: Fast, lightweight emotion detection

### Geometric Models (geo_main.py, geo_stats.py)
- **Tool**: MediaPipe Face Landmarks
- **Detection**: Facial landmarks analysis
- **Approach**: Rule-based using lip curvature, eyebrow position, etc.
- **Use case**: No AI required, fast processing

### Hybrid Models (hybrid_main.py, hybrid_stats.py)
- **Tool**: PyTorch + MediaPipe combined
- **Detection**: AI probabilities enhanced with geometric boosts
- **Approach**: AI + Geometric feature weighting
- **Use case**: Balanced accuracy/speed

### Final Models (final_main.py, final_stats.py)
- **Tool**: PyTorch + MediaPipe
- **Detection**: Strict threshold validation
- **Approach**: Position requirements + emotion threshold checklist
- **Use case**: Most accurate, requires proper positioning

## Current Thresholds (Final Models)

| Emotion | Threshold |
|---------|-----------|
| Surprise | ≥ 65 |
| Happy | ≥ 30 |
| Sad | ≥ 60 |
| Fear | ≥ 20 |
| Neutral | ≥ 70 |
| Angry | ≥ 48 |

## Position Requirements (Final)

- **Distance**: 5.5-13% of screen
- **Tilt F/B**: 5.75-8.25%
- **Turn L/R**: -3 to 3%
- Emotion only displays when ALL requirements met

## Tools Used

- **PyTorch**: Neural network models
- **OpenCV**: Image processing, Haar Cascade
- **MediaPipe**: Face landmark detection
- **NumPy**: Numerical operations

## File Architecture

EmotionDetector/
├── assets/
│   ├── face_landmarker.task
│   └── emojis/
│       ├── angry.png
│       ├── fear.png
│       ├── happy.png
│       ├── neutral.png
│       ├── sad.png
│       └── surprise.png
├── models/
│   └── emotion_model.pth
├── recordings/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── ml_main.py / ml_stats.py
│   ├── geo_main.py / geo_stats.py
│   ├── hybrid_main.py / hybrid_stats.py
│   ├── final_main.py / final_stats.py
│   └── utils/
│       ├── geometric.py
│       ├── hybrid_config.py
│       └── smoothing.py
├── train_specialist.py
├── train_hybrid_ratios.py
├── continue_training.py
├── fine_tune.py
└── requirements.txt

## Running the Models

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate  # Windows

# Run any model
python src/ml_main.py     # Pure AI
python src/ml_stats.py    # AI + stats
python src/geo_main.py   # Geometric
python src/geo_stats.py  # Geometric + stats
python src/hybrid_main.py # Hybrid
python src/hybrid_stats.py# Hybrid + stats
python src/final_main.py  # Final (no stats)
python src/final_stats.py # Final + stats
```

## Controls

- **q / Q / ESC**: Quit
- **X button**: Close window