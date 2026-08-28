"""
DEPRECATED / LEGACY MODULE:
This feature extraction module (HOG + Color Histogram) was originally used for an early
RandomForest baseline. The production architecture is now an end-to-end Deep CNN (MobileNetV2)
trained via `train_model.py` and served via `cnn_model.py` with ONNX Runtime.
"""
import cv2
import numpy as np
from skimage.feature import hog

def extract_features(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize to a consistent size (models need fixed-size input)
    image = cv2.resize(image, (128, 128))
    
    # --- Color Histogram (HSV) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # --- HOG (shape/edges) ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), feature_vector=True)
    
    # Combine both into one feature vector
    combined = np.hstack([hist, hog_features])
    return combined