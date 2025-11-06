import os
import cv2
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm


DATA_PATH = "brisc2025/classification_task/train" 
AUGMENTED_PATH = "brisc2025_balanced_aug/train"
IMG_SIZE = (512, 512)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
])

def load_dataset(path):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    for cls in class_names:
        cls_folder = os.path.join(path, cls)
        for img_file in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            images.append(img)
            labels.append(cls)
    return images, labels, class_names

def visualize_distribution(labels, title="Class distribution"):
    counter = Counter(labels)
    classes = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(8,5))
    bars = plt.bar(classes, counts, color='skyblue')
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Number of images")

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, str(count), ha='center', va='bottom', fontsize=10)
    plt.show()

def save_image(img, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, filename), img)

images, labels, class_names = load_dataset(DATA_PATH)
print("Original dataset loaded")
visualize_distribution(labels, "Original dataset distribution")


counter = Counter(labels)
max_count = max(counter.values())
print(f"Target images per class: {max_count}")

for cls in class_names:
    cls_folder = os.path.join(DATA_PATH, cls)
    aug_folder = os.path.join(AUGMENTED_PATH, cls)
    os.makedirs(aug_folder, exist_ok=True)

    img_files = [f for f in os.listdir(cls_folder) if os.path.isfile(os.path.join(cls_folder, f))]

    # Save original images with progress bar
    print(f"Saving original images for class '{cls}'")
    for img_file in tqdm(img_files, desc="Original images", ncols=100):
        img_path = os.path.join(cls_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        save_image(img, aug_folder, img_file)


    num_to_generate = max_count - len(img_files)
    print(f"Class '{cls}': generating {num_to_generate} augmented images")


    for i in tqdm(range(num_to_generate), desc="Augmented images", ncols=100):
        img_file = random.choice(img_files)
        img_path = os.path.join(cls_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        augmented = transform(image=img)['image']
        base_name, ext = os.path.splitext(img_file)
        save_image(augmented, aug_folder, f"{base_name}_aug{i+1}{ext}")

_, balanced_labels, _ = load_dataset(AUGMENTED_PATH)
visualize_distribution(balanced_labels, "Balanced dataset distribution")
