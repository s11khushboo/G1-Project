# G1- Image Classifier
# Image Classification with Convolutional Neural Networks (CNN)

Live Demo: 

https://g1-project-classifier.streamlit.app 
https://maleckicoa.com/demo-apps/

---
## 🚀 Project Overview

This project implements an **Image Classifier** using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.  
The project hosts two models: a custom CNN model and a Transfer Learning CNN model

- **Objective:** Classify images into predefined categories.
- **Framework:** TensorFlow / Keras
- **Model Type:** Deep Convolutional Neural Network (Custom) + Transfer Learning Model
- **Dataset:** CIFAR-10
- **Key Features:**
  - Multi-layer CNN with increasing filter complexity
  - Batch Normalization for stable learning
  - Dropout and L2 regularization to prevent overfitting
  - Global Average Pooling for lightweight classification
  - Trained model saved for reuse/inference

---
## 🧩 Folder Structure

Cnn_model- contains a trained model and a notebook with model specification
tl model- contains a transfer learning  link in a text file to 
        - T

---

## ⚙️ Requirements

Install the necessary dependencies using:

pip install -r requirements.txt

---

## 🧩 Model Architecture

  The CNN model consists of multiple convolutional blocks followed by a dense classifier:

---


## 💾 Saving and Loading the Model

  ### Save
  
  model.save("models/cnn_trained_model.h5")

  ### Load
  
  from tensorflow.keras.models import load_model
  
  model = load_model("cnn_model/ic_cnn_model.keras")
  
  
---
## 📊 Evaluation

Results include:

Accuracy

Loss

Confusion Matrix

Classification Report

---

## 📈 Results & Visualization

Loss and Accuracy Curves

Feature Maps (via CNN visualization)


---
