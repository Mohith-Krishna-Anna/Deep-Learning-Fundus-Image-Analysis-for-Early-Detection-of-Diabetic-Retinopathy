from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
from src.dataset import get_data_generators
from src.model import build_model
from src.config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR, LEARNING_RATE, EPOCHS, MODEL_DIR


import os

os.makedirs(MODEL_DIR, exist_ok=True)

train_gen, val_gen, _ = get_data_generators()

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Get class labels from training generator
classes = train_gen.classes
class_labels = np.unique(classes)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=class_labels,
    y=classes
)

class_weights = dict(zip(class_labels, class_weights))
print("Class Weights:", class_weights)


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights
)

model.save("models/dr_model.keras")

print(train_gen.class_indices)


