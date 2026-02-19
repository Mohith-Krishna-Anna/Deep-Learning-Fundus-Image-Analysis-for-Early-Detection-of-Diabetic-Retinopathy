import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "training")
VAL_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")

# Image parameters
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 5

# Training parameters
EPOCHS = 20
LEARNING_RATE = 0.0001
