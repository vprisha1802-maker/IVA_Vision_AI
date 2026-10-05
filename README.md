# 🔍 IVA Image Analysis System

A web-based Image Analysis application developed using **Python and Streamlit**.

This project performs four different image analysis techniques:

1. Viola-Jones Face Detection
2. FaceNet Face Embedding
3. Template Matching
4. DeepFace Analysis

---

## 📌 Project Description

The IVA Image Analysis System allows the user to upload an image and perform different computer vision and deep learning operations.

The application provides a simple web interface using **Streamlit**, so the user does not need to run separate Python programs for each operation.

### Workflow

Upload Image
       ↓
Viola-Jones Face Detection
       ↓
FaceNet Face Embedding
       ↓
Template Matching
       ↓
DeepFace Analysis
       ↓
Display Results

---

## 🚀 Features

### 1. Viola-Jones Face Detection

Viola-Jones is used to detect human faces in the uploaded image.

It uses a Haar Cascade classifier to detect faces.

**Output:**
- Detected face bounding boxes
- Number of faces detected

---

### 2. FaceNet

FaceNet converts a detected face into a numerical representation called a **face embedding**.

The embedding represents the important features of the face.

**Output:**
- Face embedding
- Embedding dimension
- First 10 embedding values

The FaceNet model generally produces a **512-dimensional embedding**.

---

### 3. Template Matching

Template Matching is used to find a small image or template inside the main image.

The user uploads:

- Main Image
- Template Image

The application compares the template with the main image.

**Output:**
- Matching location
- Blue bounding box
- Matching score
- Match status

---

### 4. DeepFace

DeepFace is used for facial attribute analysis.

The application analyzes the detected face and provides:

- Age
- Gender
- Dominant Emotion
- Emotion scores

---

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- TensorFlow
- TF-Keras
- FaceNet
- DeepFace
- SciPy

---

# 🔍 IVA Image Analysis System

A web-based **Image and Video Analytics (IVA)** application developed using **Python and Streamlit**.

The application performs multiple computer vision and deep learning operations on an uploaded image using:

- Viola-Jones
- FaceNet
- Template Matching
- DeepFace

---

## 🌐 Live Demo

🚀 **Streamlit Application:**

[Click here to open the IVA Image Analysis System](YOUR_STREAMLIT_APP_LINK)

> Replace `YOUR_STREAMLIT_APP_LINK` with your actual Streamlit deployment URL.

Example:

```text
https://iva-image-analysis.streamlit.app/


## 📂 Project Structure

```text
IVA 2/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── static/
│   └── uploads/
│
└── venv/