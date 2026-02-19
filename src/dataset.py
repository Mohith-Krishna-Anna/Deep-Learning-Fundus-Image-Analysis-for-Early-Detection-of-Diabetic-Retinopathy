import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR

def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    print("Using preprocessing:", train_datagen.preprocessing_function)


    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    print("Class indices:", train_gen.class_indices)


    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen
