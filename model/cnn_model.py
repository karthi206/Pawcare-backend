import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import PIL.ImageOps as ImageOps

CLASS_NAMES = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']

# ImageNet class indices 151-268 correspond to domestic dog breeds
DOG_CLASS_RANGE = range(151, 269)
# Validated thresholds for dog gate:
# In ImageNet (1000 classes, 118 dog breeds), a uniform prior gives 11.8% mass.
# We require at least 15% cumulative mass or top-1/top-5 dog classification.
DOG_CONFIDENCE_THRESHOLD = float(os.environ.get("DOG_GATE_THRESHOLD", "0.15"))
# Temperature parameter for probability calibration (T > 1.0 mitigates raw softmax overconfidence)
CALIBRATION_TEMPERATURE = float(os.environ.get("AI_CALIBRATION_TEMPERATURE", "1.2"))

# Normalization values matching ImageNet training (mean / std)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(weights_path):
    """
    Loads the ONNX version of the disease-detection model.
    'weights_path' points to the .onnx file (e.g. 'model/pawcare_model.onnx').
    Returns an onnxruntime InferenceSession.
    """
    session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
    return session


def load_general_model(weights_path="model/general_imagenet_model.onnx"):
    """
    Loads the ONNX version of the pretrained ImageNet MobileNetV2,
    used to verify if the uploaded image is indeed a dog before disease analysis.
    """
    session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
    return session


def _preprocess(image_path):
    """Preprocesses input image into NCHW normalized float32 tensor."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224), Image.BILINEAR)

    array = np.array(image).astype(np.float32) / 255.0        # HWC, 0-1
    array = (array - MEAN) / STD                                # normalize
    array = array.transpose(2, 0, 1)                            # HWC -> CHW
    array = np.expand_dims(array, axis=0).astype(np.float32)    # add batch dim -> NCHW
    return array


def _softmax(logits, temperature=1.0):
    """
    Computes numerically stable softmax with temperature scaling for probability calibration.
    temperature > 1.0 softens overconfident logits to match empirical probability.
    """
    t = max(float(temperature), 1e-6)
    scaled = (logits - np.max(logits)) / t
    e_x = np.exp(scaled)
    return e_x / np.sum(e_x)


def is_likely_dog(general_model, image_path, min_mass=DOG_CONFIDENCE_THRESHOLD, check_top_k=5):
    """
    Validates if an image contains a dog using multi-tier statistical gating:
      1. Top-1 prediction: If the top-1 ImageNet class is a dog breed, passes immediately.
      2. Top-k presence: If any dog breed appears in top-k predictions with cumulative mass >= min_mass.
      3. Global mass: If total dog probability mass across all 118 dog classes exceeds min_mass.

    Returns True if validated as a dog, False otherwise.
    """
    input_array = _preprocess(image_path)
    outputs = general_model.run(None, {"input": input_array})[0][0]
    probabilities = _softmax(outputs, temperature=1.0)

    top_indices = np.argsort(probabilities)[::-1]
    top1_class = top_indices[0]
    top_k_indices = top_indices[:check_top_k]

    # Check 1: Top-1 is directly a dog breed
    if top1_class in DOG_CLASS_RANGE:
        return True

    # Check 2: Total dog probability mass
    dog_probability_mass = float(sum(probabilities[i] for i in DOG_CLASS_RANGE))

    # Check 3: Any dog breed in top-k with non-trivial confidence
    has_dog_in_top_k = any(idx in DOG_CLASS_RANGE for idx in top_k_indices)

    if has_dog_in_top_k and dog_probability_mass >= min_mass:
        return True

    return dog_probability_mass >= min_mass


def predict_image(session, image_path, use_tta=False, temperature=CALIBRATION_TEMPERATURE):
    """
    Runs inference using the ONNX Runtime session with probability calibration.
    Returns calibrated prediction probabilities, top-2 classes, and ambiguity detection.
    """
    image = Image.open(image_path).convert('RGB')

    if not use_tta:
        input_array = _preprocess(image_path)
        logits = session.run(None, {"input": input_array})[0][0]
        probabilities = _softmax(logits, temperature=temperature)
    else:
        # Test-time augmentation (TTA)
        variants = [
            image,
            ImageOps.mirror(image),
            image.rotate(10, expand=False),
            image.rotate(-10, expand=False),
        ]

        all_probabilities = []
        for variant in variants:
            variant = variant.resize((224, 224), Image.BILINEAR)
            arr = np.array(variant).astype(np.float32) / 255.0
            arr = (arr - MEAN) / STD
            arr = arr.transpose(2, 0, 1)
            arr = np.expand_dims(arr, axis=0).astype(np.float32)

            logits = session.run(None, {"input": arr})[0][0]
            all_probabilities.append(_softmax(logits, temperature=temperature))

        probabilities = np.mean(all_probabilities, axis=0)

    # Top-2 predictions
    top2_indices = np.argsort(probabilities)[::-1][:2]
    top_prediction = CLASS_NAMES[top2_indices[0]]
    top_confidence = float(probabilities[top2_indices[0]])

    second_prediction = CLASS_NAMES[top2_indices[1]]
    second_confidence = float(probabilities[top2_indices[1]])

    is_ambiguous = (top_confidence - second_confidence) < 0.15

    return {
        "prediction": top_prediction,
        "confidence": top_confidence,
        "second_prediction": second_prediction if is_ambiguous else None,
        "second_confidence": second_confidence if is_ambiguous else None,
        "is_ambiguous": is_ambiguous,
        "all_probabilities": {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))},
    }

