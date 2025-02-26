from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2  # Optional if you want to use the cv2-based approach

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Set your constants: adjust IMG_SIZE as needed.
IMG_SIZE = 224
# Update with your injury detection classes: 0 for "Not Injured", 1 for "Injured"
CLASS_MAP = {0: 'Not Injured', 1: 'Injured'}

# Load the injury detection model (make sure the model path is correct)
model = tf.keras.models.load_model("models/injury.h5")

# -------------------------------
# Option 1: PIL-based prediction function
# -------------------------------
def predict_image_pil(image_bytes):
    """
    Preprocess the image using PIL, then predict using the loaded injury detection model.
    Returns:
      - class_label: The predicted label ("Injured" or "Not Injured").
      - img: The PIL image (used later for encoding to Base64).
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")                  # Ensure 3 color channels (RGB)
    img = img.resize((IMG_SIZE, IMG_SIZE))      # Resize to match model input
    img_array = np.array(img) / 255.0           # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    class_label = CLASS_MAP[np.argmax(prediction)]
    return class_label, img

# -------------------------------
# Option 2: OpenCV-based prediction function (alternative)
# -------------------------------
def predict_image_cv2(image_bytes):
    """
    Preprocess the image using OpenCV, then predict using the loaded injury detection model.
    Returns:
      - class_label: The predicted label ("Injured" or "Not Injured").
      - img_cv2: The OpenCV image (BGR format).
    """
    # Convert image bytes to a NumPy array and decode it using cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_cv2 = cv2.resize(img_cv2, (IMG_SIZE, IMG_SIZE))
    img_norm = img_cv2 / 255.0
    img_norm = np.expand_dims(img_norm, axis=0)

    prediction = model.predict(img_norm)
    class_label = CLASS_MAP[np.argmax(prediction)]
    return class_label, img_cv2

# -------------------------------
# Flask route to handle image uploads and predictions
# -------------------------------
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()

    # Use the PIL-based prediction function for simplicity
    class_label, img = predict_image_pil(image_bytes)

    # Convert the PIL image to a Base64 string so the frontend can display it.
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepend the Data URI scheme
    img_base64 = "data:image/jpeg;base64," + img_base64

    return jsonify({'result': class_label, 'image': img_base64})

# -------------------------------


if __name__ == '__main__':
    app.run(debug=True)
