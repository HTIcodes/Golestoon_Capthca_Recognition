import os
import cv2
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import random
import math

MODEL_PATH = r"captcha_model_h5.h5"
MASK_PATH = r"Data/mask.png"
MAPPING_PATH = r"char_to_idx.json"
TEST_IMAGES_DIR = r"Data/test"

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

if all(isinstance(v, int) for v in mapping_data.values()):
    idx_to_char = {v: k for k, v in mapping_data.items()}
else:
    idx_to_char = {int(k): v for k, v in mapping_data.items()}

print(f"Character mapping loaded ({len(idx_to_char)} classes)")

mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

def segment_captcha_hybrid(file_path, mask, size=(28,28)):
    img = cv2.imread(file_path)
    if img is None:
        print(f"Cannot read image: {file_path}")
        return []

    img[mask == 255] = (255, 255, 255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 8:
            boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[0])

    def split_wide_box(img, box, expected_splits=2):
        x, y, w, h = box
        roi = img[y:y+h, x:x+w]
        vertical_sum = np.sum(roi, axis=0)
        thresh = np.max(vertical_sum) * 0.5
        split_indices = []
        in_space = False
        for i, val in enumerate(vertical_sum):
            if val < thresh and not in_space:
                split_indices.append(i)
                in_space = True
            elif val >= thresh:
                in_space = False
        if len(split_indices) == 0:
            split_indices = np.linspace(0, w, expected_splits+1, dtype=int)[1:-1]
        new_boxes = []
        x_prev = 0
        for sx in split_indices:
            new_boxes.append((x + x_prev, y, sx - x_prev, h))
            x_prev = sx
        new_boxes.append((x + x_prev, y, w - x_prev, h))
        return new_boxes

    if len(boxes) <= 2:
        new_boxes = []
        for b in boxes:
            if b[2] > 20:
                new_boxes.extend(split_wide_box(processed, b))
            else:
                new_boxes.append(b)
        boxes = sorted(new_boxes, key=lambda b: b[0])

    letters = []
    for (x, y, w, h) in boxes:
        roi = processed[y:y+h, x:x+w]
        roi = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        letters.append(roi)
    return letters

def predict_captcha(image_path):
    image = cv2.imread(image_path)
    letters = segment_captcha_hybrid(image_path, mask)
    if len(letters) == 0:
        print("No letters detected!")
        return ""

    predicted_text = ""
    for roi in letters:
        roi_input = np.expand_dims(roi, axis=0)
        preds = model.predict(roi_input, verbose=0)
        pred_idx = np.argmax(preds)
        predicted_text += idx_to_char.get(pred_idx, "?")

    print(f"CAPTCHA: {os.path.basename(image_path)} → {predicted_text}")
    plt.figure(figsize=[15, 10])
    plt.imshow(image[...,::-1])
    plt.title(f"predicted captcha:{predicted_text}")
    plt.axis(False)
    plt.show()
    return predicted_text

def predict_random_captcha(count=4):
    test_images_path = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith('.png')]
    
    if len(test_images_path) < count:
        print(f"quantity of images in this dir is less than {count}")
        return

    random_images_path = random.sample(test_images_path, count)

    random_images = [cv2.imread(os.path.join(TEST_IMAGES_DIR, img_path)) for img_path in random_images_path]

    predicted_captchas = [predict_captcha(os.path.join(TEST_IMAGES_DIR, img_path))
                          for img_path in random_images_path]

    cols = 2
    rows = math.ceil(count / cols)

    plt.figure(figsize=(5*cols, 5*rows))
    for idx, img in enumerate(random_images):
        plt.subplot(rows, cols, idx+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"predicted capthca:{predicted_captchas[idx]}", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_random_captcha()