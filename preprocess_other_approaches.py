"""
============================================================
 File: preprocess_alternative.py
 Description:
     This script applies alternative image preprocessing techniques
     for leaf image datasets. It focuses on noise reduction,
     contrast enhancement, and data augmentation to improve
     the overall image quality for computer vision tasks.
============================================================

 Key Steps:
 ------------------------------------------------------------
 1. Loads Images:
    - Reads all leaf images from the dataset directory and
      converts them into a consistent RGB format.

 2. Standardizes Images:
    - Resizes each image to 128x128 pixels and normalizes
      pixel values to the [0, 1] range.

 3. Applies Alternative Preprocessing Techniques:
    - Median Filtering:
        Removes salt-and-pepper noise from the image.
    - Bilateral Filtering:
        Smooths images while preserving edges and boundaries.
    - Adaptive Histogram Equalization (CLAHE):
        Enhances local contrast and highlights fine details.
    - Gamma Correction:
        Adjusts brightness and contrast to improve image clarity.
    - Random Resized Cropping:
        Applies random crops and resizes to augment data diversity.

 4. Splits the Data:
    - Divides the processed dataset into training, validation,
      and testing subsets.

 5. Saves Preprocessed Images:
    - Exports all processed images to a separate output folder
      for training and analysis.

 6. Visualizes Results:
    - Displays intermediate steps (median, bilateral, gamma)
      and final processed images for clear visual comparison.
 ------------------------------------------------------------
"""
!pip install opencv-python-headless matplotlib scikit-learn albumentations tqdm

from google.colab import drive
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A

# Mount Google Drive (if using Colab)
drive.mount('/content/drive')

DATASET_DIR = "/content/drive/MyDrive/cv-dataset/Leaves"
OUTPUT_DIR = "/content/drive/MyDrive/cv-dataset/Leaves_preprocessed_alt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X, y = [], []
print("Loading and applying alternative preprocessing...")

for img_name in tqdm(os.listdir(DATASET_DIR), desc="Processing Images"):
    img_path = os.path.join(DATASET_DIR, img_name)
    if not (img_name.lower().endswith(('.jpg', '.png', '.jpeg'))):
        continue
    try:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0

        # Alternative preprocessing techniques

        # 1️⃣ Median filtering (remove salt-and-pepper noise)
        img_median = cv2.medianBlur((img * 255).astype(np.uint8), 3)

        # 2️⃣ Bilateral filtering (preserve edges while smoothing)
        img_bilateral = cv2.bilateralFilter((img * 255).astype(np.uint8), 9, 75, 75)

        # 3️⃣ Adaptive Histogram Equalization (CLAHE for each channel)
        lab = cv2.cvtColor(img_bilateral, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 4️⃣ Gamma correction (brighten or darken image)
        gamma = 1.5  # >1 makes darker, <1 makes brighter
        img_gamma = np.power(img_clahe / 255.0, 1.0 / gamma)
        img_gamma = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)

        # 5️⃣ Random crop (data augmentation)
        aug = A.RandomResizedCrop(128, 128, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0)
        img_aug = aug(image=img_gamma)['image']

        X.append(img_aug)
        y.append("leaf")
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print(f"\nTotal preprocessed images: {len(X)}")

X = np.array(X)
y = np.array(y)

# Train-validation split
if len(X) > 0:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
else:
    raise ValueError("No images found. Check dataset path or format.")

# Save preprocessed images
print("Saving alternative preprocessed images...")
for i, img in enumerate(X_train):
    out_path = os.path.join(OUTPUT_DIR, f"alt_leaf_{i}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"Alternative preprocessed images saved to: {OUTPUT_DIR}")

# ==========================
# 📊 Visualization Examples
# ==========================
sample = X_train[0]

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(sample)
plt.title("Final Processed Image")
plt.axis('off')

# Show intermediate stages for visualization
img = cv2.imread(os.path.join(DATASET_DIR, os.listdir(DATASET_DIR)[0]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))

median = cv2.medianBlur(img, 3)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
gamma_corrected = np.clip(np.power(median / 255.0, 1.0 / 1.5) * 255, 0, 255).astype(np.uint8)

plt.subplot(1, 4, 2)
plt.imshow(median)
plt.title("Median Filter")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(bilateral)
plt.title("Bilateral Filter")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(gamma_corrected)
plt.title("Gamma Correction")
plt.axis('off')

plt.show()