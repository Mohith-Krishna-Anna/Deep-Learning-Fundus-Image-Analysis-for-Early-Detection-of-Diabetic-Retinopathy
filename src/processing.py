import cv2
import numpy as np

def preprocess_image(img_path, img_size=224):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (img_size, img_size))

    # Normalize
    img = img / 255.0

    return img
