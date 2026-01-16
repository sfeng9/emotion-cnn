import cv2
import torch
import torchvision.transforms as transforms
from model import EmotionCNN
import os
import random
import numpy as np
from PIL import Image

# -------------------------
# Settings
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']
MODEL_PATH = "emotion_cnn.pth"
MEMES_DIR = "memes"

# -------------------------
# Load CNN Model
# -------------------------
model = EmotionCNN(num_classes=len(EMOTIONS)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# Transform for model input
# -------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -------------------------
# Load Haar Cascade for face detection
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------
# Start webcam
# -------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        # -------------------------
        # Preprocess and predict
        # -------------------------
        face_tensor = transform(face_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(face_tensor)
            emotion_idx = torch.argmax(output, dim=1).item()
            emotion = EMOTIONS[emotion_idx]

        # -------------------------
        # Draw bounding box + label
        # -------------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # -------------------------
        # Load a random meme for this emotion
        # -------------------------
        meme_folder = os.path.join(MEMES_DIR, emotion)
        if os.path.exists(meme_folder) and os.listdir(meme_folder):
            meme_path = random.choice(os.listdir(meme_folder))
            meme_img = cv2.imread(os.path.join(meme_folder, meme_path))
            # Resize meme to small window
            meme_img = cv2.resize(meme_img, (200, 200))

            # Overlay meme on top-right corner
            frame[0:200, frame.shape[1]-200:frame.shape[1]] = meme_img

    # Display webcam feed + meme
    cv2.imshow("Webcam + Cat Meme", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
