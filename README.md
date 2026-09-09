# AIML_4_PlantDiseaseDetection_byte
1. Install dependencies
2. Download the PlantVillage dataset
3. Load the trained model
4. Provide a leaf image
5. Run prediction
6. Model returns disease class and confidence
7. # Plant Disease Detection

## Overview

This project implements an image classification model for detecting
plant diseases from leaf images using the PlantVillage dataset and
MobileNetV2 transfer learning.

## Dataset

Dataset: PlantVillage

Images: approximately 54,306

Classes: 38

Plant species: 14

Dataset Source:
https://github.com/spMohanty/PlantVillage-Dataset

## Model

Model: MobileNetV2

Input Size: 224 x 224

Transfer Learning: ImageNet pretrained weights

## Dataset Preparation

- Image resizing
- Data augmentation
- Training/validation/test split
- TensorFlow preprocessing

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results

Add your actual test accuracy here.

Test Accuracy: XX.XX%

## Per-Class Performance

See:

results/classification_report.csv

## Sample Predictions

The repository contains 10 sample inference images in:

results/inference_samples/

Each image contains:

- Actual label
- Predicted label
- Prediction confidence

## How to Run

Install dependencies:

pip install -r requirements.txt

Run inference:

python src/inference.py path/to/leaf.jpg

## Model Files

plant_disease_model.keras

class_names.json

## Notebook

Plant_Disease_Detection.ipynb

## Dataset Citation

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016).
Using Deep Learning for Image-Based Plant Disease Detection.
Frontiers in Plant Science, 7, 1419.
