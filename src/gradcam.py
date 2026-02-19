import tensorflow as tf
import numpy as np
import cv2
import os

# -------- GradCAM Function --------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


# -------- Main Execution --------
if __name__ == "__main__":

    # Paths
    MODEL_PATH = "models/dr_model.keras"
    IMAGE_PATH = "sample.png"   # keep image in project root
    OUTPUT_PATH = "gradcam_output.png"

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # IMPORTANT: update this if needed
    LAST_CONV_LAYER = "top_conv"  # works for EfficientNet

    # Load & preprocess image
    img = cv2.imread(IMAGE_PATH)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img / 255.0, axis=0)

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)

    # Convert heatmap to RGB
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save output
    cv2.imwrite(OUTPUT_PATH, superimposed)

    print("Grad-CAM generated successfully → gradcam_output.png")
