import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import time


# Register the custom Cast layer
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


# Register the custom objects
custom_objects = {"Cast": CastLayer}

# Load the improved model with custom objects
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model("waste_classifier_best.h5")

# Define category mappings
# This should match the categories from your training data
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

# Set matching image size to model - UPDATED TO 192x192 to match model expectations
IMG_SIZE = 192


# Ensure img_cap directory exists for saving images
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Create img_cap directory if it doesn't exist
ensure_dir("img_cap")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Process frames less frequently to reduce load
frame_skip = 1
count = 0

# Flag to indicate if we need to take a photo
take_photo = False


def classify_image(image):
    """Classify the given image and return results"""
    # Preprocess the image for classification
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


def show_classification_popup(image, category, waste_type, confidence, save_path):
    """Display a popup with classification results that can be closed"""
    # Create a copy for display
    display_image = image.copy()

    # Resize if the image is too large
    h, w = display_image.shape[:2]
    max_height = 600
    if h > max_height:
        scale = max_height / h
        display_image = cv2.resize(display_image, (int(w * scale), max_height))

    # Add a semi-transparent overlay for text background
    overlay = display_image.copy()
    h, w = display_image.shape[:2]
    cv2.rectangle(overlay, (10, 10), (w - 10, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)

    # Display title
    cv2.putText(display_image, "Waste Classification Result",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display category
    cv2.putText(display_image, f"Category: {category}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Color-code the waste type
    if waste_type == "Recyclable":
        color = (0, 255, 0)  # Green
    elif waste_type == "Hazardous":
        color = (0, 0, 255)  # Red
    else:
        color = (255, 165, 0)  # Orange for General Waste

    cv2.putText(display_image, f"Type: {waste_type}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display confidence
    conf_text = f"Confidence: {confidence:.2f}"
    cv2.putText(display_image, conf_text,
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display save path information at the bottom
    cv2.putText(display_image, f"Image saved to: {save_path}",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_image, "Press any key to close this window",
                (w // 2 - 160, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Create a named window and show the image
    cv2.namedWindow("Classification Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Classification Result", display_image)

    # Wait for any key press to close the window
    cv2.waitKey(0)
    cv2.destroyWindow("Classification Result")


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera. Check camera connection.")
            break

        count += 1
        if count % frame_skip != 0:
            # Show the frame with instructions
            info_frame = frame.copy()
            cv2.putText(info_frame, "Press 'c' to capture and classify waste",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_frame, "Press 'q' to quit",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Waste Classifier", info_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                take_photo = True
            continue

        # Check if we need to take a photo
        if take_photo:
            # Generate timestamp and paths
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"img_cap/waste_{timestamp}.jpg"

            # Save the captured image
            cv2.imwrite(img_path, frame)
            print(f"Image saved to {img_path}")

            try:
                # Classify the image
                category, waste_type, confidence = classify_image(frame)

                # Show classification in a popup window
                show_classification_popup(frame, category, waste_type, confidence, img_path)
            except Exception as e:
                print(f"Error during classification: {str(e)}")
                error_frame = frame.copy()
                cv2.putText(error_frame, "Classification Error",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(error_frame, str(e)[:50],
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if len(str(e)) > 50:
                    cv2.putText(error_frame, str(e)[50:100],
                                (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Error", error_frame)
                cv2.waitKey(3000)
                cv2.destroyWindow("Error")

            # Reset flag
            take_photo = False

        # Display the frame with instructions
        info_frame = frame.copy()
        cv2.putText(info_frame, "Press 'c' to capture and classify waste",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_frame, "Press 'q' to quit",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Waste Classifier", info_frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            take_photo = True
except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    # Release resources
    print("Releasing camera and closing windows...")
    cap.release()
    cv2.destroyAllWindows()