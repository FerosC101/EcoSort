from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid
import time

# Initialize Flask app
app = Flask(__name__)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Scan', 'uploads')
SCAN_DIR = os.path.join(BASE_DIR, 'Scan')
TEMPLATES_DIR = os.path.join(BASE_DIR)

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_FOLDER'] = TEMPLATES_DIR

# Serve static files from root
@app.route('/')
def index():
    """
    Load the welcome/entry page first when the app starts
    """
    return send_from_directory(BASE_DIR, 'Entry page/EntryPage.html')

@app.route('/enter_app')
def enter_app():
    """
    Redirect to the scan page when 'Enter App' button is clicked
    """
    return redirect('/scan')

# Original route - will now only be accessed after clicking "Enter App"
@app.route('/scan')
def scan():
    """Serve the scan page"""
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

# Dictionary of custom objects to pass when loading model
CUSTOM_OBJECTS = {
    'Cast': CastLayer,
    # Add any other custom objects if needed
}

def try_load_model_with_savedmodel(model_path):
    """Attempt to load a model regardless of format (TF SavedModel or H5)"""
    print("Attempting to load model...")

    # First try loading as SavedModel
    try:
        model_dir = model_path
        if model_path.endswith('.h5'):
            model_dir = model_path.replace('.h5', '')

        if os.path.isdir(model_dir):
            print(f"Trying to load as SavedModel from directory: {model_dir}")
            model = tf.keras.models.load_model(model_dir, custom_objects=CUSTOM_OBJECTS)
            print("Successfully loaded as SavedModel")
            return model
    except Exception as e:
        print(f"Failed to load as SavedModel: {e}")

    # Then try loading as H5
    try:
        print(f"Trying to load as H5 file: {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        print("Successfully loaded as H5 file")
        return model
    except Exception as e:
        print(f"Failed to load as H5: {e}")

    # If all else fails, try loading with experimental_io_device
    try:
        print("Trying with experimental_io_device")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        )
        print("Successfully loaded with experimental_io_device")
        return model
    except Exception as e:
        print(f"Failed with experimental_io_device: {e}")

    raise Exception("Could not load model in any format")

# Use the model path from the debug output
model_path = os.path.join(BASE_DIR, 'MachineLearning', 'waste_classifier_best.h5')

print(f"Looking for model at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

# Attempt to find the model file with various paths
model_paths_to_try = [
    model_path,
    model_path.replace('.h5', ''),  # Try directory for SavedModel format
    os.path.join(BASE_DIR, 'waste_classifier_best.h5'),
    os.path.join(BASE_DIR, 'model'),
    os.path.join(BASE_DIR, 'saved_model')
]

# Load model with custom objects if model file exists
model = None
for path in model_paths_to_try:
    if os.path.exists(path):
        try:
            model = try_load_model_with_savedmodel(path)
            print(f"Model loaded successfully from {path}!")
            break
        except Exception as e:
            print(f"Error loading model from {path}: {str(e)}")

if model is None:
    print("Model file not found. Please make sure it exists before running the application.")

# Define waste type categories with colors (BGR format) and examples
WASTE_CATEGORIES = {
    "recyclable": {
        "color": (0, 255, 0),  # Green
        "description": "Can be recycled and processed into new items",
        "examples": "cardboard, clothes, glass, metal, paper, plastic, shoes"
    },
    "biodegradable": {
        "color": (0, 255, 255),  # Yellow
        "description": "Organic matter that decomposes naturally",
        "examples": "biological waste"
    },
    "hazardous": {
        "color": (0, 0, 255),  # Red
        "description": "Dangerous to health or environment",
        "examples": "batteries"
    },
    "residual": {
        "color": (128, 128, 128),  # Gray
        "description": "Waste that can't be reused or composted",
        "examples": "general trash"
    }
}

# Updated categories list
CATEGORIES = [
    "battery",  # hazardous
    "biological",  # biodegradable
    "brown-glass",  # recyclable
    "cardboard",  # recyclable
    "clothes",  # recyclable
    "green-glass",  # recyclable
    "metal",  # recyclable
    "paper",  # recyclable
    "plastic",  # recyclable
    "shoes",  # recyclable
    "trash",  # residual
    "white-glass"  # recyclable
]

# Map categories to waste types
CATEGORY_TO_WASTE_TYPE = {
    # Recyclable Waste
    "cardboard": "recyclable",
    "clothes": "recyclable",
    "green-glass": "recyclable",
    "white-glass": "recyclable",
    "brown-glass": "recyclable",
    "metal": "recyclable",
    "paper": "recyclable",
    "plastic": "recyclable",
    "shoes": "recyclable",

    # Biodegradable Waste
    "biological": "biodegradable",

    # Hazardous Waste
    "battery": "hazardous",

    # Residual Waste
    "trash": "residual"
}

# Create index to category mapping
IDX_TO_CATEGORY = {i: category for i, category in enumerate(CATEGORIES)}

IMG_SIZE = 224 # Updated to match the new model requirements

def preprocess_image(image):
    # Resize and convert color
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize
    img = img.astype("float32") / 255.0
    # Expand dims to match model input
    img = np.expand_dims(img, axis=0)  # Now shape is (1, 224, 224, 3)
    return img


def classify_image(image):
    if model is None:
        return "unknown", "residual", 0.0

    # Preprocess
    img = preprocess_image(image)

    # Run the model
    raw_pred = model.predict(img, verbose=0)

    # If your model has multiple outputs, pick the first one.
    # If it's just a single array, this is a no-op.
    if isinstance(raw_pred, list):
        pred = raw_pred[0]
    else:
        pred = raw_pred

    # Now ensure it’s a flat 1D numpy array
    pred = np.asarray(pred).flatten()  # shape (12,) if you have 12 classes

    # Pick your class
    predicted_class_idx = np.argmax(pred)
    confidence = float(pred[predicted_class_idx])

    category = IDX_TO_CATEGORY.get(predicted_class_idx, "unknown")
    waste_type = CATEGORY_TO_WASTE_TYPE.get(category, "residual")

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

            # Get additional waste type information
            waste_info = WASTE_CATEGORIES.get(waste_type, {"description": "Unknown"})

            return jsonify({
                'success': True,
                'category': category,
                'waste_type': waste_type,
                'description': waste_info["description"],
                'examples': waste_info.get("examples", ""),
                'confidence': float(confidence),
                'image_url': f'/uploads/{filename}'
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/webcam')
def webcam_classify():
    """Serve the webcam classification page"""
    return send_from_directory(SCAN_DIR, 'webcam.html')

def start_webcam():
    """Function to start webcam classification standalone application"""
    # Open webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Variables for prediction rate limiting
    last_prediction_time = 0
    prediction_cooldown = 0.5  # seconds between predictions
    current_result = None

    # Flag for prediction mode
    continuous_mode = False

    print("Webcam started. Press 'q' to quit, 'c' to toggle continuous prediction mode, 'p' to predict once.")
    print("Place waste items in front of the camera for classification.")

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break

        # Create a copy of the frame for display
        display_frame = frame.copy()

        # Add instructions to the frame
        cv2.putText(display_frame, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'c' to toggle continuous mode", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'p' to predict once", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display mode status
        mode_text = "Continuous Mode: ON" if continuous_mode else "Continuous Mode: OFF"
        cv2.putText(display_frame, mode_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Handle prediction based on mode
        current_time = time.time()
        should_predict = False

        if continuous_mode and (current_time - last_prediction_time) > prediction_cooldown:
            should_predict = True

        # Make prediction if needed
        if should_predict:
            try:
                category, waste_type, confidence = classify_image(frame)
                current_result = {
                    "category": category,
                    "waste_type": waste_type,
                    "confidence": confidence
                }
                last_prediction_time = current_time
            except Exception as e:
                print(f"Prediction error: {e}")
                current_result = None

        # Display results if available
        if current_result:
            # Get waste type info
            waste_type = current_result["waste_type"]
            waste_info = WASTE_CATEGORIES.get(waste_type, {"color": (255, 255, 255), "description": "Unknown"})

            # Create border around the frame
            border_size = 20
            bordered_frame = cv2.copyMakeBorder(
                display_frame,
                border_size, border_size, border_size, border_size,
                cv2.BORDER_CONSTANT,
                value=waste_info["color"]
            )

            # Display results on the frame
            # Category and waste type
            cv2.putText(bordered_frame, f"Category: {current_result['category']}",
                        (border_size + 10, bordered_frame.shape[0] - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, waste_info["color"], 2)

            # Add appropriate emoji for waste type
            emoji = "♻️" if waste_type == "recyclable" else "🍃" if waste_type == "biodegradable" else "☣️" if waste_type == "hazardous" else "🚮"
            cv2.putText(bordered_frame, f"Waste Type: {emoji} {waste_type.upper()}",
                        (border_size + 10, bordered_frame.shape[0] - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, waste_info["color"], 2)

            # Confidence score
            cv2.putText(bordered_frame, f"Confidence: {current_result['confidence']:.2f}",
                        (border_size + 10, bordered_frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, waste_info["color"], 2)

            # Show the bordered frame
            cv2.imshow("Waste Classification", bordered_frame)
        else:
            # Show the regular frame if no prediction
            cv2.imshow("Waste Classification", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            continuous_mode = not continuous_mode
            print(f"Continuous mode: {'ON' if continuous_mode else 'OFF'}")
        elif key == ord('p') and not continuous_mode:
            try:
                category, waste_type, confidence = classify_image(frame)
                current_result = {
                    "category": category,
                    "waste_type": waste_type,
                    "confidence": confidence
                }
                last_prediction_time = current_time
            except Exception as e:
                print(f"Prediction error: {e}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam application closed.")

if __name__ == '__main__':
    # Add a command-line argument parser to support both server and webcam modes
    import argparse

    parser = argparse.ArgumentParser(description='Waste Classification Application')
    parser.add_argument('--webcam', action='store_true', help='Run in webcam mode instead of server mode')
    args = parser.parse_args()

    if args.webcam:
        # Run in webcam mode
        if model is None:
            print("WARNING: Model not loaded. Application will return default values for classifications.")
            print("Please make sure the model file exists at the correct location and restart the application.")
        else:
            print("Starting webcam classification mode...")
            start_webcam()
    else:
        # Run in server mode
        if model is None:
            print("WARNING: Model not loaded. Application will return default values for classifications.")
            print("Please make sure the model file exists at the correct location and restart the application.")
        else:
            print("Application is ready! Access it at http://127.0.0.1:5000")

        app.run(debug=True, port=5000)