# FABRIC-DEFECT-DETECTION-FOR-E-COMMERCE-QUALITY-CONTROL

# Fabric Defect Detection Using CNN (ResNet50 + Coordinate Attention)

## Overview
This project aims to automate fabric defect detection by classifying
fabric images into **defective** and **non-defective** categories.
The system is designed to assist quality control in the e-commerce
clothing industry by reducing manual inspection.

## Objectives
- Develop an automated fabric inspection system using deep learning
- Improve accuracy and efficiency in identifying defective fabrics
- Support sustainable production by reducing defective product delivery

## Tools & Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## Dataset
The project uses the **Warping Fabric Defect Detection (WFDD)** dataset
from Kaggle, containing **4,101 high-resolution images** across four
fabric types:
- Grey cloth
- Grid cloth
- Pink flower cloth
- Yellow cloth

Images are labeled as **good** or **defective**, with defect types such as
contaminated, flecked, line, and string (mainly in grey cloth).

> Dataset is not included in this repository due to size constraints.

## Methodology
1. **Dataset Reorganization**
   - Original dataset reorganized into two folders: `good` and `defect`
   - Enables automatic labeling using Keras `flow_from_directory()`

2. **Preprocessing**
   - Images resized to **224 × 224**
   - Pixel normalization to range [0, 1]
   - Train-validation split: **80% training, 20% validation**

3. **Data Augmentation**
   - Random rotation (±10°)
   - Zoom up to 10%
   - Horizontal flipping
   - Applied only to training data to improve robustness

## 🧠Model Architecture
The model is based on **ResNet50**, pre-trained on ImageNet, with:
- Coordinate Attention (CA) block for enhanced feature focus
- Global Average Pooling
- Dense + ReLU activation
- Dropout for regularization
- Sigmoid output layer for binary classification

The model was trained using:
- Adam optimizer
- Binary cross-entropy loss
- Class weighting to handle class imbalance
- Early stopping to prevent overfitting

Fine-tuning was applied by unfreezing base layers and retraining with a
lower learning rate to improve performance.

## Results
The trained model demonstrates effective classification of defective
and non-defective fabric images, showing improved sensitivity to defect
instances through class balancing and attention mechanisms.

## 🚀 What I Learned
- Image preprocessing and augmentation techniques
- Transfer learning with ResNet50
- Attention mechanisms in CNNs
- Handling class imbalance in real-world datasets
- Model fine-tuning and evaluation
