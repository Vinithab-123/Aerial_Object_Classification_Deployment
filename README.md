# Aerial_Object_Classification_Deployment
Deep Learning solution for binary classification of aerial images as Bird or Drone (MobileNetV2)

# 🦅 Aerial Object Classification AI (Bird vs. Drone)

## 🌟 Project Overview

This project implements a Deep Learning solution for a critical task in aerial surveillance: **distinguishing between birds and drones** in real-time aerial imagery.

The goal was to train a high-performance classification model and deploy it as a user-friendly web application using **Streamlit**.

### 🎯 Key Objective

To build a robust binary classifier capable of achieving high accuracy in distinguishing between aerial images of **Birds** and **Drones**.

---

## 🚀 Model Performance & Results

The project compared two models: a Custom Convolutional Neural Network (CNN) and a Transfer Learning approach using MobileNetV2.

| Model Architecture | Final Validation Accuracy | Role |
| :--- | :--- | :--- |
| **MobileNetV2 (Transfer Learning)** | **97.21%** | Chosen Production Model |
| Custom CNN | [Your Custom CNN Accuracy]% | Baseline Comparison |

The **MobileNetV2** model was selected for its superior generalization and performance, achieving over **97% accuracy** on the unseen validation dataset.

---

## 💻 Technical Stack

* **Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow 2.x / Keras
* **Web Framework:** Streamlit
* **Core Libraries:** NumPy, Pillow (PIL)
* **Version Control:** Git, Git LFS (for large model files)

---

## 📂 Repository Structure

The project is organized into the following directories:

| Directory | Content |
| :--- | :--- |
| **`src/`** | Contains the main application script (`app.py`). |
| **`models/`** | Stores the final trained Keras models (`best_mobilenet_model.keras`). |
| **`notebooks/`** | Contains Jupyter Notebooks detailing data preprocessing, EDA, and model training/comparison. |
| **`requirements.txt`** | Lists all required Python packages for reproducibility. |

---

## ⚙️ Local Setup and Installation

Follow these steps to get a local copy of the project running on your machine.

### 1. Clone the Repository

Since the repository contains large model files, ensure you have **Git LFS** installed on your system before cloning.

```bash
# 1. Clone the repository
git clone [https://github.com/Vinithab-123/Aerial_Object_Classification_Deployment.git](https://github.com/Vinithab-123/Aerial_Object_Classification_Deployment.git)

# 2. Navigate to the project directory
cd Aerial_Object_Classification_Deployment
