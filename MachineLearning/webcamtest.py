import os
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import json

# ─── 1. Custom Cast layer ──────────────────────────────────────────────────────
class Cast(Layer):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype
    def call(self, inputs):
        return tf.cast(inputs, self._dtype)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'dtype': self._dtype})
        return cfg

# ─── 2. Load model & mappings ─────────────────────────────────────────────────
model = load_model('EcoSort_models/final.h5', custom_objects={'Cast': Cast})
with open('../models/category_map.json') as f:
    category_map = json.load(f)
with open('../models/group_map.json') as f:
    group_map = json.load(f)
idx_to_class = {v:k for k,v in category_map.items()}
idx_to_group = {int(k):v for k,v in group_map.items()}

# ─── 3. Setup capture directories ──────────────────────────────────────────────
BASE_CAPTURE_DIR = 'captures'
os.makedirs(BASE_CAPTURE_DIR, exist_ok=True)
for cls in idx_to_class.values():
    os.makedirs(os.path.join(BASE_CAPTURE_DIR, cls), exist_ok=True)

# ─── 4. Prediction + capture parameters ────────────────────────────────────────
IMG_SIZE   = 224
THRESHOLD  = 0.8   # only capture if model is ≥80% confident
cap        = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Preprocess for model ───────────────────────────────────────────────────
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_input = img.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # ── Run inference ──────────────────────────────────────────────────────────
    det_preds, grp_preds = model.predict(img_input, verbose=0)
    det_idx, det_conf = int(np.argmax(det_preds)), np.max(det_preds)
    grp_idx, grp_conf = int(np.argmax(grp_preds)), np.max(grp_preds)

    # ── Map to labels ─────────────────────────────────────────────────────────
    cat = idx_to_class.get(det_idx, 'unknown')
    grp = idx_to_group.get(det_idx, 'unknown')

    # ── Overlay live label ────────────────────────────────────────────────────
    text = f"{cat.upper()} ({grp}) {det_conf*100:.1f}%"
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # ── Capture if above threshold ────────────────────────────────────────────
    if det_conf >= THRESHOLD and cat != 'unknown':
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{ts}_{cat}_{int(det_conf*100)}.jpg"
        out_dir = os.path.join(BASE_CAPTURE_DIR, cat)
        cv2.imwrite(os.path.join(out_dir, fname), frame)
        print(f"Captured: {os.path.join(out_dir, fname)}")

    # ── Show frame & handle quit ──────────────────────────────────────────────
    cv2.imshow("EcoSort Live Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
