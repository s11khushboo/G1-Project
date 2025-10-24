# G1-Project
# Image Classification with Convolutional Neural Networks (CNN)

This project implements an **Image Classifier** using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.  
The model is trained to classify images into multiple categories and includes features such as **Batch Normalization**, **Dropout Regularization**, and **Global Average Pooling** for improved performance and generalization.

---
## 🚀 Project Overview

- **Objective:** Train a CNN to classify images into predefined categories.
- **Framework:** TensorFlow / Keras
- **Model Type:** Deep Convolutional Neural Network (Custom)
- **Dataset:** Any image dataset (e.g., CIFAR-10, custom dataset)
- **Key Features:**
  - Multi-layer CNN with increasing filter complexity
  - Batch Normalization for stable learning
  - Dropout and L2 regularization to prevent overfitting
  - Global Average Pooling for lightweight classification
  - Trained model saved for reuse/inference

---

##💾 **Saving and Loading the Model**

# Save
model.save("models/cnn_trained_model.h5")

# Load
from tensorflow.keras.models import load_model
model = load_model("models/cnn_trained_model.h5")
## 🧩 Model Architecture

The CNN model consists of multiple convolutional blocks followed by a dense classifier:

---

## ⚙️ Requirements

Install the necessary dependencies using:

pip install -r requirements.txt

---
## 📊 Evaluation
python evaluate.py --data path/to/testset --model saved_model.h5

Results include:

Accuracy

Loss

Confusion Matrix

Classification Report

---

##📈 Results & Visualization 

Loss and Accuracy Curves

Feature Maps (via CNN visualization)

Grad-CAM Heatmaps (optional)

---
