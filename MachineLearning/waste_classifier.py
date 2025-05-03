import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

# Set memory growth to prevent GPU memory explosion
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Optional: Enable mixed precision for better performance on compatible hardware
try:
    from tensorflow.keras.mixed_precision import set_global_policy

    set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except:
    print("Mixed precision not supported in this TensorFlow version")

# Dataset Path
base_dir = "trashnet"


# Create class weights to handle imbalanced data
def calculate_class_weights(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))

    total = sum(class_counts.values())
    class_weights = {i: total / (len(class_counts) * count)
                     for i, (cls, count) in enumerate(class_counts.items())}
    return class_weights


class_weights = calculate_class_weights(base_dir)

# More balanced and efficient data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,  # Reduced from 40
    width_shift_range=0.2,  # Reduced from 0.3
    height_shift_range=0.2,  # Reduced from 0.3
    shear_range=0.2,  # Reduced from 0.3
    zoom_range=0.2,  # Reduced from 0.3
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # More constrained
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# Reduced image size to decrease memory usage
IMG_SIZE = 192  # Reduced from 224

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,  # Smaller batch size
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,  # Smaller batch size
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_indices = train_generator.class_indices
category_mapping = {v: k for k, v in class_indices.items()}
print(f"Classes detected: {category_mapping}")

# Build model using transfer learning instead of training from scratch
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model initially

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(class_indices), activation='softmax')
])

model.summary()

optimizer = Adam(learning_rate=0.001)  # Higher initial learning rate for transfer learning
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "waste_classifier_best_og.h5",
    save_best_only=True,
    monitor="val_accuracy",
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,  # Reduced from 5
    min_lr=0.00001,
    verbose=1
)

# Train with frozen base model first
print("Phase 1: Training only top layers...")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # Fewer epochs for initial training
    callbacks=[early_stopping, checkpoint, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# Unfreeze the last few layers of the base model for fine-tuning
print("Phase 2: Fine-tuning the model...")
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
optimizer = Adam(learning_rate=0.0001)  # Much lower learning rate for fine-tuning
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Continue training with unfrozen layers
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    class_weight=class_weights,
    verbose=1,
    initial_epoch=len(history1.history['loss'])  # Continue from where we left off
)

# Combine histories for plotting
history = history1
for key in history.history:
    history.history[key].extend(history2.history[key])

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_history.png')


def predict_and_analyze(model, generator):
    x, y_true = next(generator)
    y_pred = model.predict(x)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    class_accuracy = {}
    for class_id in range(len(class_indices)):
        mask = y_true_classes == class_id
        if np.sum(mask) > 0:
            class_accuracy[category_mapping[class_id]] = np.mean(y_pred_classes[mask] == class_id)

    print("\nClass-wise accuracy:")
    for cls, acc in class_accuracy.items():
        print(f"{cls}: {acc:.2f}")

    return x[:5], y_true_classes[:5], y_pred_classes[:5]


# Analyze model performance on validation data
test_images, true_labels, pred_labels = predict_and_analyze(model, val_generator)

# Reset the generator to ensure we get all validation samples
val_generator.reset()

# Calculate predictions for all validation data
steps = len(val_generator)
y_true = []
y_pred = []

# Predict in smaller batches to avoid memory issues
for i in range(steps):
    x_batch, y_batch = next(val_generator)
    batch_pred = model.predict(x_batch)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(batch_pred, axis=1))

# Create confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=list(category_mapping.values()))

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

print("\nModel training complete! Best model saved as waste_classifier_best.h5")

# Convert to TFLite for more efficient inference
print("Converting model to TFLite format for more efficient inference...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('waste_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved as waste_classifier.tflite")


# Function to use when deploying the model to detect all waste types
def predict_waste_type(image_path, model, category_mapping):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]

    result = {
        "waste_type": category_mapping[class_idx],
        "confidence": float(confidence),
        "recyclable": category_mapping[class_idx] in ["plastic", "paper", "glass", "metal"],
        "all_probabilities": {category_mapping[i]: float(prob) for i, prob in enumerate(prediction[0])}
    }

    return result


# Function to use the TFLite model for inference (more efficient)
def predict_waste_type_tflite(image_path, tflite_model_path, category_mapping):
    from tensorflow.keras.preprocessing import image

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Process image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Check if the model expects float32
    if input_details[0]['dtype'] == np.float32:
        img_array = img_array.astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get prediction results
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]

    result = {
        "waste_type": category_mapping[class_idx],
        "confidence": float(confidence),
        "recyclable": category_mapping[class_idx] in ["plastic", "paper", "glass", "metal"],
        "all_probabilities": {category_mapping[i]: float(prob) for i, prob in enumerate(prediction)}
    }

    return result