import tensorflow as tf
import os
import cv2
import numpy as np
import json
import time

# Configure GPU memory growth to avoid memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_model_safely(model_path):
    """Load a Keras model with various fallback methods."""
    print(f"Attempting to load model from: {model_path}")

    # Define a custom Cast layer class to handle mixed precision
    class Cast(tf.keras.layers.Layer):
        def __init__(self, dtype=None, **kwargs):
            super(Cast, self).__init__(**kwargs)
            self.dtype_to_cast = dtype

        def call(self, inputs):
            return tf.cast(inputs, self.dtype_to_cast)

        def get_config(self):
            config = super(Cast, self).get_config()
            config.update({"dtype": self.dtype_to_cast})
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    # Try multiple loading methods
    try:
        # First try with custom objects
        print("Attempting to load with custom Cast layer...")
        custom_objects = {'Cast': Cast}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully with custom objects!")
        return model
    except Exception as e1:
        print(f"First attempt failed: {e1}")

        try:
            # Second attempt: try loading with compile=False
            print("Attempting to load with compile=False...")
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded with compile=False")
            return model
        except Exception as e2:
            print(f"Second attempt failed: {e2}")

            # Try the simplest approach - it's a last resort
            try:
                print("Final attempt: trying to load model...")
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
                print("Model loaded with basic loader")
                return model
            except Exception as e3:
                print(f"Final attempt failed: {e3}")
                raise Exception("Could not load model with any method")


def main():
    """Main function to run the webcam waste classifier."""
    # Look for model in the current directory
    model_dir = '.'

    # Find the model file - look for both best and final versions
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5') and ('waste_classifier' in f)]

    if not model_files:
        print(
            "Error: No model file found. Please make sure waste_classifier_best.h5 or waste_classifier_final.h5 exists.")
        return

    model_path = os.path.join(model_dir, model_files[0])
    print(f"Found model: {model_path}")

    # Load the model
    try:
        model = load_model_safely(model_path)

        # Test model with a dummy input
        print("Testing model with dummy input...")
        dummy_input = np.random.random((1, 224, 224, 3))
        dummy_output = model.predict(dummy_input, verbose=0)
        print(f"Model prediction works! Output type: {type(dummy_output)}")
    except Exception as e:
        print(f"Error loading or testing model: {e}")
        return

    # Define basic waste categories for demonstration
    category_mapping = {
        0: "battery", 1: "biological", 2: "brown-glass", 3: "cardboard",
        4: "clothes", 5: "green-glass", 6: "metal", 7: "paper",
        8: "plastic", 9: "shoes", 10: "trash", 11: "white-glass"
    }

    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam waste classifier running!")
    print("Press 'c' to capture and classify")
    print("Press 'q' to quit")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break

        # Add overlay with instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 40),
                      (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, "Press 'c' to capture and classify, 'q' to quit",
                    (20, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the frame
        cv2.imshow('Waste Classification', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

        # If 'c' is pressed, capture and process the frame
        elif key == ord('c'):
            # Show processing message
            processing_frame = frame.copy()
            cv2.putText(processing_frame, "Processing...",
                        (frame.shape[1] // 2 - 80, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Waste Classification', processing_frame)
            cv2.waitKey(1)  # Update display

            try:
                # Preprocess image
                img = cv2.resize(frame, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = img / 255.0  # Normalize
                img = np.expand_dims(img, axis=0)  # Add batch dimension

                # Get prediction
                start_time = time.time()
                prediction = model.predict(img, verbose=0)
                end_time = time.time()

                # Get detailed prediction (first output of the model)
                if isinstance(prediction, list) and len(prediction) > 0:
                    detailed_pred = prediction[0]
                    detailed_class_idx = np.argmax(detailed_pred[0])
                else:
                    detailed_class_idx = np.argmax(prediction[0])

                category = category_mapping.get(detailed_class_idx, f"Unknown ({detailed_class_idx})")

                # Draw results
                result_frame = frame.copy()
                overlay = result_frame.copy()
                cv2.rectangle(overlay, (0, result_frame.shape[0] - 120),
                              (result_frame.shape[1], result_frame.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)

                cv2.putText(result_frame, f"Classified as: {category}",
                            (20, result_frame.shape[0] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.putText(result_frame, f"Processing time: {(end_time - start_time) * 1000:.1f} ms",
                            (20, result_frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                # Save the image with prediction
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                os.makedirs("waste_captures", exist_ok=True)
                img_path = os.path.join("waste_captures", f"waste_{timestamp}.jpg")
                cv2.imwrite(img_path, frame)

                # Save prediction text
                txt_path = os.path.join("waste_captures", f"waste_{timestamp}_prediction.txt")
                with open(txt_path, 'w') as f:
                    f.write(f"Material: {category}\n")

                cv2.putText(result_frame, f"Saved to {img_path}",
                            (20, result_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Show result for 3 seconds
                cv2.imshow('Waste Classification', result_frame)
                cv2.waitKey(3000)

            except Exception as e:
                print(f"Error processing image: {e}")
                error_frame = frame.copy()
                cv2.putText(error_frame, f"Error: {str(e)[:30]}...",
                            (20, error_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Waste Classification', error_frame)
                cv2.waitKey(3000)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam waste classifier closed.")


if __name__ == "__main__":
    main()