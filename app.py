from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

@app.route("/test")
def test():
    pass

@app.route("/register", methods=["GET", "POST"])
def register():
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # For now simple check (you can connect DB later)
        email = request.form["email"]
        password = request.form["password"]

        if email and password:
            return redirect(url_for("prediction"))

    return render_template("login.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
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
            prediction = CLASS_NAMES[predicted_class]

    return render_template("prediction.html",
                           prediction=prediction,
                           confidence=round(float(confidence * 100), 2) if confidence else None,
                           image_path=image_path)

@app.route("/logout")
def logout():
    return render_template("logout.html")


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
