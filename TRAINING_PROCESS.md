# Emotion Detection Model: Training & Fine-Tuning Process

This document outlines the systematic process used to train and fine-tune the Emotion Detection model, progressing from a base accuracy of ~59.12% toward the target of 70% using the FER2013 dataset.

## 1. Model Architecture Evolution
Initially, a custom Convolutional Neural Network (CNN) was used. To improve performance, we transitioned to a more robust **ResNet-18** architecture:
- **Pre-trained Backbone:** Leveraged ResNet-18 weights pre-trained on ImageNet (Transfer Learning).
- **Grayscale Optimization:** Modified the first convolutional layer to accept 1-channel grayscale input while preserving pre-trained feature extraction capabilities by averaging weights across channels.
- **Custom Classifier Head:** Replaced the final fully connected layer with a high-performance head:
  - Dropout layers (0.4) for regularization.
  - 512-unit dense layer with ReLU activation.
  - Batch Normalization for training stability.
  - Final 7-unit output layer for emotion classification.

## 2. Advanced Data Augmentation
To prevent overfitting and improve generalization on the FER2013 dataset, we implemented industry-standard augmentation techniques:
- **Spatial Transforms:** Random Horizontal Flips, Rotations (up to 15°), and Affine translations/scaling.
- **Color Jitter:** Subtle brightness and contrast adjustments to simulate different lighting conditions.
- **Random Erasing:** Masking small parts of the image to force the model to learn multiple facial features.
- **Normalization:** Standardized inputs using the dataset's calculated mean and standard deviation.

## 3. Training Strategy
The training process was executed in multiple stages to ensure stable convergence:

### Stage A: Base Training (`src/train.py`)
- **Optimizer:** `AdamW` with a weight decay of `5e-3` for better regularization than standard Adam.
- **Loss Function:** `CrossEntropyLoss` with:
  - **Class Weights:** Computed from the dataset distribution to handle class imbalance (e.g., more 'Happy' images than 'Disgust').
  - **Label Smoothing (0.1):** Prevents the model from becoming over-confident and improves generalization.
- **Learning Rate Schedule:** `CosineAnnealingWarmRestarts` to navigate complex loss landscapes and escape local minima.

### Stage B: Fine-Tuning & Optimization
When the model reached performance plateaus, specific fine-tuning scripts were used:
- **Classifier-Only Fine-Tuning:** Freezing the convolutional "backbone" and only training the final layers to refine classification without destroying learned features.
- **Safe Continued Training:** Using ultra-conservative learning rates (e.g., `0.00005`) and `ReduceLROnPlateau` to squeeze out final percentage gains.
- **GPU Acceleration:** Leveraging CUDA (RTX 5070) for significantly faster iteration cycles.

## 4. Validation & Best Model Selection
- **Validation Split:** 15-20% of the data was held out for validation.
- **Checkpointing:** Only models that achieved a "New Best" validation accuracy were saved to `models/emotion_model.pth`.
- **Early Stopping:** Implemented to stop training if no improvement was seen for a set number of epochs (patience), preventing wasted compute and overfitting.

## 5. Summary of Progress
1. **Starting Point:** Custom CNN reaching ~59.12%.
2. **Phase 1:** Transition to ResNet-18 architecture.
3. **Phase 2:** Refinement of Augmentation and Class Weights.
4. **Current Goal:** 70% Accuracy via high-stability fine-tuning.
