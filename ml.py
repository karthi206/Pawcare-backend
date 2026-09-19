import os
import sys

from config import MODEL_DIR

sys.path.append(MODEL_DIR)

from cnn_model import load_model, predict_image, load_general_model, is_likely_dog, load_ood_reference  # noqa: E402

# Points to the ML v2 dual-output model (logits + pooled features), needed
# for Mahalanobis OOD detection. Replaces the old single-output
# pawcare_model.onnx. See PawCare ML v2 roadmap Step 13/14.
MODEL_PATH = os.path.join(MODEL_DIR, 'pawcare_mobilenetv2_with_features.onnx')
GENERAL_MODEL_PATH = os.path.join(MODEL_DIR, 'general_imagenet_model.onnx')
CLASS_MEANS_PATH = os.path.join(MODEL_DIR, 'class_means.npy')
COV_INV_PATH = os.path.join(MODEL_DIR, 'cov_inv.npy')

model = load_model(MODEL_PATH)
general_model = load_general_model(GENERAL_MODEL_PATH)
class_means, cov_inv = load_ood_reference(CLASS_MEANS_PATH, COV_INV_PATH)
# CONFIDENCE_THRESHOLD is applied inside predict_image() (defaults to the
# data-justified 0.7 from cnn_model.py); no longer computed here.

__all__ = [
    "model", "general_model", "class_means", "cov_inv",
    "predict_image", "is_likely_dog",
]
