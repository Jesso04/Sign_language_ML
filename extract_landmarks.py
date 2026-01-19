import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

FIXED_FRAMES = 60

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_landmarks(video_path):
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

    return frames


INPUT_ROOT = r"D:\Sign language project\Sign_language_ML_in_Visual_studio_code\Subset_Vedios"
OUTPUT_ROOT = r"D:\Sign language project\Sign_language_ML_in_Visual_studio_code\Landmarks"

for label in os.listdir(INPUT_ROOT):
    input_dir = os.path.join(INPUT_ROOT, label)
    output_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(os.listdir(input_dir), desc=f"Processing {label}"):
        if not file.endswith(".mp4"):
            continue

        video_path = os.path.join(input_dir, file)
        data = extract_landmarks(video_path)

        if data is None:
            continue
        

        out_file = file.replace(".mp4", ".npy")
        np.save(os.path.join(output_dir, out_file), data)
