import numpy as np
import onnxruntime as ort
from PIL import Image
import PIL.ImageOps as ImageOps

CLASS_NAMES = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']

DOG_CLASS_RANGE = range(151, 269)  # ImageNet class indices 151-268 are dog breeds
DOG_CONFIDENCE_THRESHOLD = 0.05    # sum of probability mass in dog classes must exceed this

# Same normalization values used during training (ImageNet mean/std)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(weights_path):
    """
    Loads the ONNX version of the disease-detection model.
    'weights_path' should point to the .onnx file (e.g. 'model/pawcare_model.onnx').
    Returns an onnxruntime InferenceSession instead of a PyTorch model.
    """
    session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
    return session


def load_general_model():
    """
    Dog-detection gate is disabled for memory reasons (as before).
    Kept as a no-op so existing imports in app.py don't break.
    """
    return None


def _preprocess(image_path):
    """Replicates the old torchvision transform (resize -> tensor -> normalize) using PIL/numpy."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224), Image.BILINEAR)

    array = np.array(image).astype(np.float32) / 255.0        # HWC, 0-1
    array = (array - MEAN) / STD                                # normalize
    array = array.transpose(2, 0, 1)                            # HWC -> CHW
    array = np.expand_dims(array, axis=0).astype(np.float32)    # add batch dim -> NCHW
    return array


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def is_likely_dog(general_model, image_path):
    """Disabled — always returns True so nothing blocks uploads. Kept for interface compatibility."""
    return True


def predict_image(session, image_path, use_tta=False):
    """
    Runs inference using the ONNX Runtime session.
    'session' is the object returned by load_model().
    """
    image = Image.open(image_path).convert('RGB')

    if not use_tta:
        input_array = _preprocess(image_path)
        outputs = session.run(None, {"input": input_array})[0][0]
        probabilities = _softmax(outputs)
    else:
        # Test-time augmentation: average predictions across a few variants.
        variants = [
            image,
            ImageOps.mirror(image),           # horizontal flip
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

            outputs = session.run(None, {"input": arr})[0][0]
            all_probabilities.append(_softmax(outputs))

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
    }
