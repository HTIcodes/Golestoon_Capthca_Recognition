import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import csv

CAPTCHA_IMG_DIR = r"Data/train"
CAPTCHA_CSV = r"Data/train_data.csv"
OUTPUT_DIR = r"captcha_segment"
MASK = r"Data/mask.png"

png_files = glob.glob(CAPTCHA_IMG_DIR + "/*.png")
captcha_images = [cv2.imread(file) for file in png_files]

def clean_captcha_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)

    df = df[~df["text"].str.contains(r'[\$\*\\]')]

    df = df[df["text"].str.len() == 5]

    df.reset_index(drop=True, inplace=True)
    return df

captcha_df = clean_captcha_dataset(CAPTCHA_CSV)

mask = cv2.imread(r"C:\Users\Mahyar\Desktop\Captcha\mask.png", cv2.IMREAD_GRAYSCALE)

def process_and_save_letters_hybrid(file_path, label, mask, output_dir=OUTPUT_DIR, size=(28,28)):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(file_path)
    if img is None:
        print(f"⚠️ cannot read image: {file_path}")
        return False

    img[mask == 255] = (255, 255, 255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

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

    def split_wide_box(img, box, expected_chars):
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
        if len(split_indices) < expected_chars-1:
            split_indices = np.linspace(0, w, expected_chars+1, dtype=int)[1:-1]
        new_boxes = []
        x_prev = 0
        for sx in split_indices:
            new_boxes.append((x + x_prev, y, sx - x_prev, h))
            x_prev = sx
        new_boxes.append((x + x_prev, y, w - x_prev, h))
        return new_boxes

    if len(boxes) < len(label):
        new_boxes = []
        deficit = len(label) - len(boxes)
        sorted_boxes = sorted(boxes, key=lambda b: b[2], reverse=True)
        for b in sorted_boxes:
            if deficit > 0 and b[2] > 20:
                splits = split_wide_box(processed, b, 2)
                new_boxes.extend(splits)
                deficit -= (len(splits)-1)
            else:
                new_boxes.append(b)
        boxes = sorted(new_boxes, key=lambda b: b[0])

    if len(boxes) != len(label):
        ys, xs = np.where(processed == 255)
        points = np.column_stack((xs, ys)).astype(np.float32)
        if len(points) >= len(label):
            K = len(label)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
            _, k_labels, _ = cv2.kmeans(points, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
            boxes_kmeans = []
            for i in range(K):
                cluster_points = points[k_labels.ravel() == i]
                if cluster_points.size == 0:
                    continue
                x_min, y_min = np.min(cluster_points, axis=0).astype(int)
                x_max, y_max = np.max(cluster_points, axis=0).astype(int)
                boxes_kmeans.append((x_min, y_min, x_max-x_min, y_max-y_min))
            boxes = sorted(boxes_kmeans, key=lambda b: b[0])

    if len(boxes) != len(label):
        print(f"❌ mismatch: {file_path} (boxes={len(boxes)}, label={len(label)})")
        return False

    for i, (x, y, w, h) in enumerate(boxes):
        roi = processed[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        resized = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        out_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_{i}_{label[i]}.png"
        cv2.imwrite(os.path.join(output_dir, out_name), resized)

    return True

safe_files = [f for f in captcha_df["filename"].values if os.path.exists(os.path.join(CAPTCHA_IMG_DIR, f))]

miss_matched = 0
for file_name in safe_files:
    file_path = os.path.join(CAPTCHA_IMG_DIR, file_name)
    label = captcha_df.loc[captcha_df["filename"] == file_name, "text"].values[0]

    ok = process_and_save_letters_hybrid(file_path, label, mask)
    if not ok:
        miss_matched += 1
        continue
print(f"count miss_match: {miss_matched}")