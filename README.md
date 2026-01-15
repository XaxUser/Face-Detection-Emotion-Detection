# Face & Emotion Detection System 🎭

A hybrid Computer Vision project that combines **Viola-Jones** (for face detection) and a custom **PyTorch CNN** (for emotion classification).

The system identifies faces in an image and classifies their expression into three categories: **Happy**, **Sad**, or **Neutral**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Face%20Detection-green)

## 📂 Project Structure
```text
├── models/             # Contains the trained .pth weights
├── notebooks/          # Jupyter notebook for training and analysis
├── src/                # Source code
│   └── model.py        # The CNN architecture definition
├── main.py             # Script to run detection on an image
└── requirements.txt    # Dependencies