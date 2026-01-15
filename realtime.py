import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from src.model import EmotionCNN

# --- CONFIGURATION ---
MODEL_PATH = 'models/emotion_model.pth' # Ensure you downloaded this from Colab!
CLASS_NAMES = ['happy', 'neutral', 'sad']
# Auto-detect device (Use CPU if no NVIDIA GPU is found)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = EmotionCNN(num_classes=len(CLASS_NAMES))
    # map_location ensures it loads on CPU even if trained on Colab GPU
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        print("Did you download 'emotion_model.pth' from Colab and put it in the 'models' folder?")
        exit()
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded!")
    return model

def run_realtime_emotion():
    # 1. Load Model & Setup
    model = load_model()
    
    inference_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])

    # 2. Setup Camera
    # '0' is usually the default webcam. If you have multiple, try 1.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    # Load Face Detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print("🎥 Starting Real-Time Emotion Detection... Press 'q' to quit.")

    while True:
        # A. Grab a single frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # B. Detect Faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # C. Process Each Face
        for (x, y, w, h) in faces:
            # Crop
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess (Convert to PIL -> Tensor)
            # We use cv2.cvtColor because PIL expects RGB, OpenCV gives BGR
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_face)
            img_tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)

            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            emotion = CLASS_NAMES[pred.item()]
            score = conf.item() * 100

            # Draw (Color Logic)
            if emotion == 'happy': color = (0, 255, 0)     # Green
            elif emotion == 'sad': color = (0, 0, 255)     # Red
            else: color = (0, 255, 255)                    # Yellow

            # Draw Box & Label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{emotion} {score:.0f}%", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # D. Show the Frame
        cv2.imshow('Real-Time Emotion AI', frame)

        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_emotion()