from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid
import time
import json

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Scan', 'uploads')
SCAN_DIR = os.path.join(BASE_DIR, 'Scan')
ABOUT_DIR = os.path.join(BASE_DIR, 'about_section')
MAPPING_DIR = os.path.join(BASE_DIR, 'Mapping')
DISPOSAL_DIR = os.path.join(BASE_DIR, 'DisposalGuide')
ML_DIR = os.path.join(BASE_DIR, 'MachineLearning', 'EcoSort_models')

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

# Dictionary of custom objects to pass when loading model
CUSTOM_OBJECTS = {
    'Cast': CastLayer,
}

# Paths to model files
model_path = os.path.join(ML_DIR, 'final.h5')
category_map_path = os.path.join(ML_DIR, 'category_map.json')
group_map_path = os.path.join(ML_DIR, 'group_map.json')

# Initialize model and mappings
model = None
idx_to_class = {}
idx_to_group = {}

try:
    # Load model
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    # Load category and group mappings
    with open(category_map_path) as f:
        category_map = json.load(f)
        idx_to_class = {v: k for k, v in category_map.items()}

    with open(group_map_path) as f:
        group_map = json.load(f)
        idx_to_group = {int(k): v for k, v in group_map.items()}

    print("Model and mappings loaded successfully!")
except Exception as e:
    print(f"Error loading model or mappings: {str(e)}")

# Define waste type categories
WASTE_CATEGORIES = {
    "recyclable": {
        "color": (0, 255, 0),
        "description": "Can be recycled and processed into new items",
        "disposal": "Place in the recycling bin. Make sure items are clean and dry.",
        "examples": "cardboard, clothes, glass, metal, paper, plastic, shoes"
    },
    "biodegradable": {
        "color": (0, 255, 255),
        "description": "Organic matter that decomposes naturally",
        "disposal": "Place in compost bin or designated organic waste container.",
        "examples": "food scraps, garden waste"
    },
    "hazardous": {
        "color": (0, 0, 255),
        "description": "Dangerous to health or environment",
        "disposal": "Take to special collection points. Do not mix with regular trash.",
        "examples": "batteries, chemicals, electronics"
    },
    "residual": {
        "color": (128, 128, 128),
        "description": "Waste that can't be reused or composted",
        "disposal": "Place in general waste bin. Try to minimize this type of waste.",
        "examples": "non-recyclable plastics, contaminated materials"
    }
}

IMG_SIZE = 224

def preprocess_image(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(image):
    if model is None:
        return "unknown", "residual", 0.0

    img = preprocess_image(image)
    det_preds, grp_preds = model.predict(img, verbose=0)

    det_idx = int(np.argmax(det_preds))
    det_conf = float(np.max(det_preds))
    category = idx_to_class.get(det_idx, "unknown")

    grp_idx = int(np.argmax(grp_preds))
    grp_conf = float(np.max(grp_preds))
    waste_type = idx_to_group.get(det_idx, "residual")

    return category, waste_type, det_conf

# Route for home page
@app.route('/')
def home():
    return send_from_directory(os.path.join(BASE_DIR, 'Entry page'), 'EntryPage.html')

# Route for about page
@app.route('/about')
def about():
    return send_from_directory(ABOUT_DIR, 'about.html')

# Route for scan page
@app.route('/scan')
def scan():
    return send_from_directory(SCAN_DIR, 'scan.html')

# Route for maps page
@app.route('/maps')
def maps():
    return send_from_directory(MAPPING_DIR, 'index.html')

# Route for disposal guide
@app.route('/disposal')
def disposal():
    return send_from_directory(DISPOSAL_DIR, 'disposal.html')

# Route to enter the app from entry page
@app.route('/enter_app')
def enter_app():
    return redirect('/scan')

# API endpoint for image classification
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
        filename = secure_filename(f"{uuid.uuid4()}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Could not read image file'}), 400

            category, waste_type, confidence = classify_image(image)
            waste_info = WASTE_CATEGORIES.get(waste_type, {
                "color": (255, 255, 255),
                "description": "Unknown",
                "disposal": "Please check local waste disposal guidelines",
                "examples": "Unknown"
            })

            return jsonify({
                'success': True,
                'category': category,
                'waste_type': waste_type,
                'description': waste_info["description"],
                'disposal': waste_info["disposal"],
                'examples': waste_info.get("examples", ""),
                'confidence': float(confidence),
                'image_url': f'/uploads/{filename}'
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Serve files from Scan directory
@app.route('/Scan/<path:filename>')
def serve_scan_files(filename):
    return send_from_directory(SCAN_DIR, filename)

# Serve files from AboutWithFooter directory
@app.route('/AboutWithFooter/<path:filename>')
def serve_about_files(filename):
    return send_from_directory(ABOUT_DIR, filename)

# Serve files from Mapping directory
@app.route('/Mapping/<path:filename>')
def serve_mapping_files(filename):
    return send_from_directory(MAPPING_DIR, filename)

# Serve files from DisposalGuide directory
@app.route('/DisposalGuide/<path:filename>')
def serve_disposal_files(filename):
    return send_from_directory(DISPOSAL_DIR, filename)

# Serve static files from root
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)