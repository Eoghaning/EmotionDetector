# Emotion Detection: Model Training & Hybrid System Architecture

## 1. Core Model Architecture (ResNet-18)
Transitioned from custom CNNs to a **ResNet-18** backbone. 
- **MILESTONE REACHED:** **67.00% Accuracy** on ML Model (Final Version).
- **6-Emotion optimization:** System focuses on **Angry, Fear, Happy, Sad, Surprise, Neutral**. 

## 2. Training & Specialization
- **Stage A (Base):** Standard transfer learning.
- **Stage B (Final ML):** Reached 67% using `continue_training_gpu.py` with RTX 5070.
- **Saving Policy:** Verified highest-accuracy weights are saved in `models/emotion_model.pth`.

## 3. Geometric & Hybrid Optimization
- **Geometric Model:** Rules-based landmarks used for physical ground-truth.
- **Hybrid Ratio Training:** `train_hybrid_ratios.py` finds the optimal balance between AI and Geometry.
- **Dataset:** Features extracted into local `.npy` files for ratio optimization.

## 4. Security & Repository Rules
- **DO NOT COMMIT** any `.npy` files or `.pth` weight files to GitHub.
- **DO NOT COMMIT** `.env` or configuration files with sensitive paths.
- **Training updates** must be logged exclusively in this document.

## 5. Summary of Methods
- **`src/ml_main.py`**: Pure AI (Final Version: 67%+).
- **`src/geo_main.py`**: Pure Geometry.
- **`src/hybrid_main.py`**: Optimized Hybrid System.

## 6. Project Status (April 2026)
- **Status:** Step 1 (ML Model Training) is **COMPLETE**.
- **Performance:** High stability across 6 target emotions.
- **Next Milestone:** Finalizing Hybrid Ratio optimization.
