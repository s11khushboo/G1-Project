# G1- Image Classifier
# Image Classification with Convolutional Neural Networks (CNN)

Live Demos: 

https://g1-project-classifier.streamlit.app 
https://maleckicoa.com/demo-apps/

---
## 🚀 Project Overview

This project implements an **Image Classifier** using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.  
The project hosts two models: a custom CNN model and a Transfer Learning CNN model

- **Objective:** Classify images into predefined categories.
- **Framework:** TensorFlow / Keras
- **Model Type:** Deep Convolutional Neural Network (Custom) & Transfer Learning Model
- **Dataset:** CIFAR-10
- **Key Features:**
  - Multi-layer CNN with increasing filter complexity
  - Batch Normalization for stable learning
  - Dropout and L2 regularization to prevent overfitting
  - Global Average Pooling for lightweight classification
  - Trained model saved for reuse/inference

---
## 🧩 Folder Structure

Cnn_model- contains a trained custom Keras CNN model and a IPYNB notebook with a specification on how to train the model
tl model- contains a text file with a link to the Keras Transfer learning CNN model
experiment_model - Contains various model versions that were made/trained during the development

---

## ⚙️ Requirements

Install the necessary dependencies using:
pip install -r requirements.txt

---

## 🧩 Model Architecture

  The Custom CNN model: 
      - consists of multiple convolutional blocks followed by a dense classifier (see: cnn_model/image_classifier_cnn.ipynb)
      
  The Transfer Learning model: 
      - Base model is MobileNetV2 followed with a few custom AvgPooling and Dense layers ( see: tl_model/CIFAR-10- Transfer Learning Winner Model.ipynb )

---


## 💾 How to run the models

  ### Approch 1
  
  run the `model.predict` method on preprocessed images 
  (see: cnn_model/image_classifier_cnn.ipynb &  tl_model/CIFAR-10- Transfer Learning Winner Model.ipynb )
  

  ### Approach 2
  
  Feed a raw image into the Web application
  (https://g1-project-classifier.streamlit.app 
  https://maleckicoa.com/demo-apps/)
  
  
---
## 📊 Model Results

  ### Approch 1
  
  The model results from the IPYNB notebooks include: Accuracy, Loss, Confusion Matrix, Classification Report and performance visualisations
  
  ### Approach 2
  
  For a given image, the Web application returns the image class with a probability
    
