import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset import get_data_generators
from keras.models import load_model

_, _, test_gen = get_data_generators()

model = load_model("models/dr_model.keras")


predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
