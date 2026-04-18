# Face2Emoji

## The Story Behind Face2Emoji

This whole project started with a simple question: can we make a computer understand what someone is feeling just by looking at their face? Turns out, yes—but getting it to work well took some creative problem-solving.

### Why Python?

Python became the obvious choice for this project for good reason. The ecosystem for computer vision and machine learning in Python is unmatched. **OpenCV** gives us battle-tested image processing capabilities, **PyTorch** makes neural network development accessible, and the whole ecosystem integrates seamlessly. When you're prototyping vision systems that need to iterate fast, Python's readability and the vast library support make it the only real choice.

### The Dataset: FER2013

The training data comes from the **FER2013 dataset**—a collection of 35,887 grayscale 48x48 face images labeled with 7 basic emotions. It's the standard benchmark for facial expression recognition and presents real challenges: the images are small, inconsistent in lighting, and sometimes ambiguous even for humans. We used this data to train our custom ResNet-18 model from scratch.

### The Tools We Used

- **Python**: The backbone of everything—clean, expressive, and packed with libraries
- **OpenCV**: Real-time webcam capture, face detection, image preprocessing, and visual overlays
- **PyTorch**: Custom ResNet-18 implementation for emotion classification
- **MediaPipe**: Google's landmark detection library for extracting 468 facial reference points
- **NumPy**: Efficient numerical operations for geometric calculations

### The Architecture: ResNet-18

We experimented with custom CNNs early on but switched to **ResNet-18** as our backbone. It's lightweight enough to run in real-time on a CPU while still having enough capacity to learn meaningful features from 48x48 grayscale images. The pretrained structure helps with feature extraction even when training from scratch on emotion data.

### The Big Breakthrough: 67% Accuracy

After months of training, GPU acceleration, and careful hyperparameter tuning, we hit **67% accuracy** on our trained model. That might not sound groundbreaking, but consider this: human agreement on emotion labeling is only about 70%. We're essentially at human level on a challenging dataset.

### The Hybrid Approach

Here's where things got interesting. Instead of just relying on pixel patterns, we built a second system that analyzes **facial geometry**—measuring mouth aspect ratio, lip curvature, eyebrow positions, and other physical landmarks using MediaPipe detection. Then we combined both systems: the trained model gives a probability distribution, and the geometric detector adds its vote based on what it sees in the face structure.

This hybrid system revealed that certain emotions have reliable physical signatures:
- **Surprise** and **Fear**: Wide-open mouth, raised eyebrows
- **Happy**: Curved-up lips, exposed teeth pattern
- **Sad**: Downturned mouth corners, droopy features
- **Angry**: Tightened jaw, lowered inner eyebrows

### The 75% Goal

We're currently pushing toward 75% accuracy on the hybrid system. The plan involves:
1. **Continued training** with GPU to squeeze more performance from the base model
2. **Ratio tuning** to find the perfect mathematical balance between pattern recognition and geometric signals
3. **Specialist training** for tricky emotion pairs that the model keeps confusing (looking at you, Fear vs Surprise)

---

# The Four Model Types

## 1. ML Model (`src/ml_main.py`)

**Pure pattern recognition, no geometric analysis.**

This is the foundation of everything—a custom ResNet-18 neural network trained on FER2013 that runs inference on face crops. It uses OpenCV's Haar Cascade classifier to detect faces, extracts the region of interest, and feeds it through the trained model. The output is a probability distribution over 6 emotions (Angry, Fear, Happy, Sad, Surprise, Neutral).

This model runs entirely on pixel patterns and learned features. No handcrafted rules, no geometric measurements—just raw statistical inference.

**Best for:** Understanding what the trained model "sees" in isolation, with no outside influence.

---

## 2. Geo Model (`src/geo_main.py`)

**Pure geometric analysis, no trained model involved.**

This model completely ignores the trained neural network and relies entirely on handcrafted rules. It uses MediaPipe Face Landmarker to extract 468 facial landmarks, then calculates various ratios and angles:
- **Mouth Aspect Ratio (MAR)**: How open is the mouth?
- **Lip Curvature**: Are the corners up or down?
- **Eyebrow Position**: Raised, lowered, or neutral?
- **Facial Symmetry**: Is the expression lopsided?

Based on these measurements, a rule-based system determines the most likely emotion. It's surprisingly good at detecting extreme expressions but struggles with subtle or mixed emotions that don't have clear geometric signatures.

**Best for:** Testing the "physical" side of expressions. Great for understanding which emotions have obvious structural patterns.

---

## 3. Hybrid Model (`src/hybrid_main.py`)

**An automatic combination of Model 1 (ML) and Model 2 (Geometric).**

This system runs both the trained neural network AND the geometric detector simultaneously, then combines their outputs using mathematical balancing. When the geometric detector identifies a strong physical signal (like an obvious smile or wide-open mouth), it mathematically boosts the corresponding probability from the trained model.

The hybrid approach is more robust than either system alone because:
- The trained model catches subtle expressions the geometric rules might miss
- The geometric detector provides consistency for extreme, obvious expressions
- They compensate for each other's weaknesses through automated weighting

The hybrid system includes:
- A **dashboard** showing scores from all three branches (ML, Geo, Hybrid combined)
- A **truth table** for conflict resolution (e.g., Sad beats Angry when both signals fire)
- **Automatic weighting** that combines both models' predictions

**Best for:** General real-time use. The hybrid approach handles edge cases better than pure pattern recognition.

---

## 4. Final Model (`src/final_main.py`)

**The Hybrid model with manual threshold tweaking on top.**

This is the same hybrid system as Model 3, but with manual refinement—essentially taking the automatic hybrid math and applying hand-tuned rules and thresholds based on real-world testing. Instead of letting the math decide everything, specific emotions have been adjusted:

- **Strict thresholds** prevent false positives (e.g., Surprise needs 95% confidence to trigger—a wide-open mouth is rarely a false positive)
- **Fixed display values** for extreme emotions (Fear always displays as 15 when detected—the physical signature of fear is distinctive enough)
- **Dynamic anger penalty** that reduces Angry confidence when Neutral indicators are present (angry faces often look blank until they suddenly don't)
- **Neutral override**: If Neutral hits 50%, it always displays as Neutral
- **Manual conflict resolution** with specific rules for emotion pairs:
  - Neutral vs anything: pick the non-neutral emotion
  - Fear vs others: Fear × 3 competes against the other emotion × 0.8

**The key difference:** The Hybrid model lets the math balance the two systems automatically. The Final model applies manual corrections on top of that hybrid base—it's the same foundation, but with human-tuned adjustments to reduce jitter and false positives based on how it performed in testing.

**Best for:** When you need minimal jitter and high confidence. This is the most refined version for real-world deployment.

---

# Current Performance (April 2026)

| Component | Accuracy |
|-----------|----------|
| Trained Model (ML) | 67.00% |
| Hybrid Combined | 66.96% |
| Target Goal | 75.00% |

The numbers are nearly identical between pure pattern recognition and hybrid—the real gains will come from better training data, longer training sessions, and specialist models for confused emotion pairs. The Final model with its manual thresholds doesn't aim to improve accuracy on paper—it aims to improve *perceived* accuracy by reducing false triggers and prediction jitter.
