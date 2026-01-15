import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.model import EmotionCNN
import sys

# CONFIG
MODEL_PATH = 'models/emotion_model.pth' # Path to your saved model
CLASS_NAMES = ['happy', 'neutral', 'sad']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = EmotionCNN(num_classes=len(CLASS_NAMES))
    # Load weights if available, otherwise warn user
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"⚠️ Warning: {MODEL_PATH} not found. Running with random weights (for testing only).")
    
    model.to(DEVICE)
    model.eval()
    return model

def predict_emotions(image_path):
    # 1. Load Model
    model = load_model()
    
    # 2. Setup Transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])

    # 3. Detect Faces (Viola-Jones)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Error: Image {image_path} not found.")
        return

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 5, minSize=(30, 30))
    
    print(f"🔍 Found {len(faces)} faces.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 4. Process Faces
    for (x, y, w, h) in faces:
        face_crop = img_rgb[y:y+h, x:x+w]
        pil_img = Image.fromarray(face_crop)
        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        emotion = CLASS_NAMES[pred.item()]
        text = f"{emotion} {conf.item()*100:.0f}%"
        
        # Draw
        if emotion == 'happy': color = (0, 255, 0)
        elif emotion == 'sad': color = (255, 0, 0)
        else: color = (255, 255, 0)
        
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(img_rgb, (x, y-25), (x+w, y), color, -1)
        cv2.putText(img_rgb, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # 5. Show
    plt.figure(figsize=(10,10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default image if none provided
        image_path = 'test_image.jpg' 
    
    predict_emotions(image_path)