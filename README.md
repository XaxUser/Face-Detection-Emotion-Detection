# Face & Emotion Detection System 🎭

A hybrid Computer Vision project that combines **Viola-Jones** (for fast face detection) and a custom **PyTorch CNN** (for emotion classification).

The system identifies faces in real-time or in static images and classifies their expression into three categories: **Happy**, **Sad**, or **Neutral**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Real%20Time-green)

## ✨ Features
* **Real-Time Detection:** Uses your webcam to detect emotions live.
* **Static Image Analysis:** Upload any photo to detect and label emotions.
* **Hybrid Architecture:** * **Locator:** OpenCV Haar Cascades (CPU-efficient).
    * **Classifier:** Custom CNN trained on FER-2013 (GPU-accelerated).

## 📂 Project Structure
```text
├── models/             # Contains the trained .pth weights
│   └── emotion_model.pth
├── notebooks/          # Jupyter notebook for training and research
├── src/                # Source code
│   └── model.py        # The CNN architecture definition
├── main.py             # Script to run detection on a static image
├── realtime.py         # Script to run detection on Webcam (Real-Time)
└── requirements.txt    # Dependencies