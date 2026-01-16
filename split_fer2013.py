import os
import pandas as pd
import numpy as np
import cv2

# Paths
CSV_PATH = "fer2013.csv"
OUTPUT_DIR = "data"

emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

usage_map = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test"
}

# Load CSV
df = pd.read_csv(CSV_PATH)

# Create directories
for split in ["train", "val", "test"]:
    for emotion in emotion_map.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, emotion), exist_ok=True)

# Convert rows to images
for idx, row in df.iterrows():
    emotion = emotion_map[row["emotion"]]
    split = usage_map[row["Usage"]]

    pixels = np.array(row["pixels"].split(), dtype="uint8")
    image = pixels.reshape(48, 48)

    filename = f"{idx}.png"
    filepath = os.path.join(OUTPUT_DIR, split, emotion, filename)

    cv2.imwrite(filepath, image)

    if idx % 5000 == 0:
        print(f"Processed {idx} images")

print("FER-2013 successfully split into folders!")
