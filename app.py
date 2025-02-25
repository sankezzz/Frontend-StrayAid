from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the model
model = tf.keras.models.load_model("cnn_model.h5")  # Ensure this is your actual model

# Function to preprocess the image
def preprocess_image(image):
    img = image.convert("RGB")  # Ensure 3 color channels (RGB)
    img = img.resize((224, 224))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    img = Image.open(io.BytesIO(image_file.read()))  # Open image
    processed_img = preprocess_image(img)  # Preprocess for the model

    # Get model prediction
    prediction = model.predict(processed_img)
    output_label = f"Detected: {np.argmax(prediction)}"  # Adjust based on your model output

    # Convert image to base64 string for frontend display
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'result': output_label, 'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
