import os
import json
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

SEGMENT_DIR = r"captcha_segment"

def load_segmented_dataset(segment_dir):
    images = []
    labels = []
    files = [f for f in os.listdir(segment_dir) if f.endswith(".png")]
    for f in files:
        img_path = os.path.join(segment_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        images.append(img)

        label = f.split("_")[-1].split(".")[0]
        labels.append(label)

    print(f"Loaded {len(images)} samples from {segment_dir}")
    return np.array(images), np.array(labels)

X, y = load_segmented_dataset(SEGMENT_DIR)

unique_chars = sorted(list(set(y)))
char_to_idx = {c: i for i, c in enumerate(unique_chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

with open("char_to_idx.json", "w", encoding="utf-8") as f:
    json.dump(char_to_idx, f, ensure_ascii=False, indent=4)
with open("idx_to_char.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_char, f, ensure_ascii=False, indent=4)

y_encoded = np.array([char_to_idx[c] for c in y])
y_onehot = to_categorical(y_encoded, num_classes=len(unique_chars))

X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(unique_chars), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val))

model.save(r"captcha_model_h5.h5")
model.save(r"captca_model_keras.keras")

print("Model trained and saved successfully!")