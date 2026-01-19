import os
import numpy as np
import json

LANDMARKS_DIR = r"D:\Sign language project\Sign_language_ML_in_Visual_studio_code\Landmarks"

X = []
y = []
label_map = {}

label_index = 0

for label_name in sorted(os.listdir(LANDMARKS_DIR)):
    label_path = os.path.join(LANDMARKS_DIR, label_name)
    if not os.path.isdir(label_path):
        continue

    label_map[label_name] = label_index

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(label_path, file))
            X.append(data)
            y.append(label_index)

    label_index += 1

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("Dataset created successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label map:", label_map)
