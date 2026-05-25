import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "breast_cancer_densenet_model.keras"
IMAGE_PATH = "test_image.jpg"  # change to your image

IMG_SIZE = 224
LAST_CONV_LAYER = "conv5_block16_concat"

CLASS_NAMES = ["Benign", "Malignant"]

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_array = np.expand_dims(img_rgb, axis=0)
img_array = preprocess_input(img_array)

# =========================
# PREDICTION
# =========================
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    label = "Malignant"
    confidence = prediction
else:
    label = "Benign"
    confidence = 1 - prediction

print("Prediction:", label)
print("Confidence:", confidence)

# =========================
# GRAD-CAM FUNCTION
# =========================
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(LAST_CONV_LAYER).output, model.output]
)

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))

# =========================
# OVERLAY HEATMAP
# =========================
heatmap = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# =========================
# DISPLAY RESULTS
# =========================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"{label} ({confidence*100:.2f}%)")
plt.axis("off")

plt.show()