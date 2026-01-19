import cv2
import numpy as np
import torch
import torch.nn as nn
import json
import mediapipe as mp
from collections import deque
from pathlib import Path

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_model.pth"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"

# -------------------------------
# LOAD LABEL MAP
# -------------------------------
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

id_to_label = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)

# -------------------------------
# MODEL (SAME AS TRAINING)
# -------------------------------
class SignLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = SignLSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -------------------------------
# MEDIAPIPE
# -------------------------------
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -------------------------------
# CONSTANTS
# -------------------------------
FIXED_FRAMES = 60
frame_buffer = deque(maxlen=FIXED_FRAMES)

# -------------------------------
# OPEN WEBCAM
# -------------------------------
cap = cv2.VideoCapture(0)

print("Webcam started. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand.landmark]
        frame_buffer.append(landmarks)

        # Predict only when buffer is full
        if len(frame_buffer) == FIXED_FRAMES:
            data = np.array(frame_buffer).reshape(1, FIXED_FRAMES, 63)
            data = torch.tensor(data, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)

            label = id_to_label[pred.item()]
            confidence = conf.item()

            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
