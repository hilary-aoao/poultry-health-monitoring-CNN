# PoultryVision: CNN-Based Disease Classification from Fecal Images

## Overview
This project builds an intelligent poultry health monitoring system that automatically
classifies poultry diseases from fecal images using deep learning. It is designed to
serve as a first-line diagnostic tool for small to medium scale poultry farmers,
particularly in regions like Nigeria and across Africa where poultry farming plays
a vital role in food security and livelihoods.

## Problem Statement
Late disease diagnosis in poultry farming often leads to the loss of entire flocks,
causing significant financial damage to farmers. This model provides an automated,
accessible and affordable solution for early disease detection directly from fecal images.

## Dataset
- **Source:** Chicken Disease Image Classification — Kaggle (allandclive)
- **Total Images:** 8,067 fecal images
- **Classes:** Healthy, Coccidiosis, Salmonella, New Castle Disease

## Model
- **Architecture:** ResNet18 with Transfer Learning
- **Pretrained on:** ImageNet
- **Approach:** Froze all base layers and replaced the final fully connected layer
  with a custom classifier for 4 classes
- **Class imbalance handled with:** Weighted CrossEntropyLoss

## Results
| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Healthy | 0.88 | 0.90 | 0.89 |
| Coccidiosis | 0.94 | 0.93 | 0.93 |
| Salmonella | 0.95 | 0.86 | 0.90 |
| New Castle Disease | 0.62 | 0.86 | 0.72 |
| **Overall Accuracy** | | | **89.22%** |

## Tech Stack
- Python
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib & Seaborn
- Pandas & NumPy

## Project Structure
poultry-health-monitoring/
│
├── poultryvision-cnn-based-disease-classification.ipynb   # Main notebook with full pipeline
├── README.md                  # Project documentation

## Key Learnings
- Transfer learning with ResNet18 achieves strong results even on small medical datasets
- Class weighting is essential when dealing with imbalanced disease datasets
- Data augmentation significantly improves generalisation on unseen images

## Future Work
- Collect more New Castle Disease images to address class imbalance at the data level
- Unfreeze deeper ResNet layers and fine tune with a lower learning rate
- Combine fecal images with physical symptom data for a multi-modal diagnostic system

## Notebook
View the full notebook on Kaggle: [PoultryVision on Kaggle]https://www.kaggle.com/code/hillivize/poultryvision-cnn-based-disease-classification
