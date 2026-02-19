from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

@app.route("/test")
def test():
    pass

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "models/dr_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [ "No DR", "Mild", "Moderate", "Severe", "Proliferative DR" ]


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            img = preprocess_image(image_path)
            preds = model.predict(img)[0] 
            confidence = np.max(preds) 
            predicted_class = np.argmax(preds)


             # Safety threshold
            if confidence < 0.4:
                prediction = f"{CLASS_NAMES[predicted_class]} (Low confidence)"
            else:
                prediction = CLASS_NAMES[predicted_class]


    return render_template("index.html",
                           prediction=prediction,
                           confidence=round(float(confidence * 100), 2) if confidence is not None else None,
                           image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
