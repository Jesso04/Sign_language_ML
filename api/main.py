import os
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
import mediapipe as mp

from fastapi import FastAPI, UploadFile, File
from tempfile import NamedTemporaryFile
from pathlib import Path

# -------------------------------------------------
# PATH SETUP (THIS FIXES ALL WINDOWS PATH ISSUES)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

LABEL_MAP_PATH = BASE_DIR / "label_map.json"
MODEL_PATH = BASE_DIR / "sign_model.pth"

# -------------------------------------------------
# LOAD LABEL MAP
# -------------------------------------------------
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

id_to_label = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)

# -------------------------------------------------
# MODEL DEFINITION (MUST MATCH TRAINING)
# -------------------------------------------------
class SignLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = SignLSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -------------------------------------------------
# MEDIAPIPE SETUP
# -------------------------------------------------
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

FIXED_FRAMES = 60

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand.landmark]
            frames.append(landmarks)

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.array(frames)

    if len(frames) > FIXED_FRAMES:
        frames = frames[:FIXED_FRAMES]
    else:
        pad = np.zeros((FIXED_FRAMES - len(frames), 21, 3))
        frames = np.vstack([frames, pad])

    # reshape to (1, 60, 63)
    frames = frames.reshape(1, FIXED_FRAMES, 63)
    return torch.tensor(frames, dtype=torch.float32)

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------
app = FastAPI(title="Sign Language to Text API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded video temporarily
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(await file.read())
        video_path = temp.name

    # Extract landmarks
    landmarks = extract_landmarks_from_video(video_path)

    # Remove temp file
    os.remove(video_path)

    if landmarks is None:
        return {
            "error": "No hand detected in the uploaded video"
        }

    # Model inference
    with torch.no_grad():
        outputs = model(landmarks)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label = id_to_label[pred.item()]

    return {
        "prediction": label,
        "confidence": float(confidence.item())
    }
