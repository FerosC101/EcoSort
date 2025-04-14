from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid

# Initialize Flask app
app = Flask(__name__)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Scan', 'uploads')
SCAN_DIR = os.path.join(BASE_DIR, 'Scan')

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Serve static files from root
@app.route('/')
def serve_index():
    return send_from_directory(SCAN_DIR, 'scan.html')


# Serve Scan directory files
@app.route('/Scan/<path:filename>')
def serve_scan_files(filename):
    return send_from_directory(SCAN_DIR, filename)


# Serve other HTML files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)


# Register custom Cast layer
class CastLayer(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float32, **kwargs):
        self.dtype_value = dtype
        super(CastLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.cast(inputs, self.dtype_value)

    def get_config(self):
        config = super(CastLayer, self).get_config()
        config.update({"dtype": self.dtype_value})
        return config


# Use the model path from the debug output
model_path = os.path.join(BASE_DIR, 'MachineLearning', 'waste_classifier_best.h5')

print(f"Looking for model at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

# Load model with custom objects if model file exists
if os.path.exists(model_path):
    try:
        custom_objects = {"Cast": CastLayer}
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None
else:
    print("Model file not found. Please make sure it exists before running the application.")
    model = None

# Category mappings
category_mapping = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}

waste_type_mapping = {
    'cardboard': 'Recyclable',
    'glass': 'Hazardous',
    'metal': 'Recyclable',
    'paper': 'Recyclable',
    'plastic': 'Recyclable',
    'trash': 'General Waste'
}

IMG_SIZE = 192


def classify_image(image):
    """Classify the given image and return results"""
    if model is None:
        return "unknown", "unknown", 0.0

    # Preprocess the image
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Model prediction
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    # Get waste type and category
    category = category_mapping[predicted_class]
    waste_type = waste_type_mapping[category]

    return category, waste_type, confidence


@app.route('/api/classify', methods=['POST'])
def handle_classification():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded file
        filename = secure_filename(f"{uuid.uuid4()}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read and classify the image
        try:
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Could not read image file'}), 400

            category, waste_type, confidence = classify_image(image)

            return jsonify({
                'success': True,
                'category': category,
                'waste_type': waste_type,
                'confidence': float(confidence),
                'image_url': f'/uploads/{filename}'
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    if model is None:
        print("WARNING: Model not loaded. Application will return default values for classifications.")
        print("Please make sure the model file exists at the correct location and restart the application.")
    else:
        print("Application is ready! Access it at http://127.0.0.1:5000")

    app.run(debug=True, port=5000)