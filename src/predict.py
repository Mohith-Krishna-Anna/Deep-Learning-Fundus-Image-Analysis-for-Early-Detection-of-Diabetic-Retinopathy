import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

MODEL_PATH = "models/dr_model.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [ "No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR" ]


def predict_image(img_path):
    if not os.path.exists(img_path):
        print("Image path does not exist")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    print("Predicted Class:", CLASS_NAMES[predicted_class])



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <image_path>")
    else:
        predict_image(sys.argv[1])
