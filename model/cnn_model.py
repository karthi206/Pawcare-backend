import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import PIL.ImageOps as ImageOps


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "demodicosis",
    "ringworm",
]

# ImageNet dog-breed class indices used by the general model.
DOG_CLASS_RANGE = range(151, 269)

DOG_CONFIDENCE_THRESHOLD = float(
    os.environ.get("DOG_GATE_THRESHOLD", "0.05")
)

# --- Calibration & rejection settings ---
# These are the values actually fitted/validated in the ML v2 Colab notebook,
# not placeholders. See PawCare ML v2 roadmap Steps 8-10.
#
# Temperature: fit via LBFGS on validation-set logits (NLL 0.2309 -> 0.1893)
CALIBRATION_TEMPERATURE = float(
    os.environ.get("AI_CALIBRATION_TEMPERATURE", "1.5525")
)

# Confidence threshold: chosen from test-set data — below this, accuracy was
# only ~75%; at/above it, accuracy was 98%+.
CONFIDENCE_THRESHOLD = float(
    os.environ.get("AI_CONFIDENCE_THRESHOLD", "0.7")
)

# OOD (out-of-distribution) threshold: Mahalanobis distance in the model's
# pooled 1280-dim feature space. Full test-set distances ranged 21.42-80.56
# (99th percentile 74.47); random noise scored 82-88. Set just above the
# 99th percentile of real images and below the observed test-set max.
OOD_THRESHOLD = float(
    os.environ.get("AI_OOD_THRESHOLD", "75.0")
)

# ImageNet normalization.
MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(weights_path):
    """
    Load the ONNX disease-detection model.

    This model has TWO outputs: 'logits' (classification logits, shape
    [1, 6]) and 'features' (pooled penultimate-layer features, shape
    [1, 1280]) used for out-of-distribution detection.
    """

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Disease model not found: {weights_path}"
        )

    return ort.InferenceSession(
        weights_path,
        providers=["CPUExecutionProvider"],
    )


def load_general_model(
    weights_path="model/general_imagenet_model.onnx"
):
    """
    Load the ONNX ImageNet model used for dog validation.
    """

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"General ImageNet model not found: {weights_path}"
        )

    return ort.InferenceSession(
        weights_path,
        providers=["CPUExecutionProvider"],
    )


def load_ood_reference(
    class_means_path="model/class_means.npy",
    cov_inv_path="model/cov_inv.npy",
):
    """
    Load the per-class feature means and inverse covariance matrix used
    for Mahalanobis-distance out-of-distribution detection.
    """

    if not os.path.exists(class_means_path):
        raise FileNotFoundError(
            f"OOD class means not found: {class_means_path}"
        )
    if not os.path.exists(cov_inv_path):
        raise FileNotFoundError(
            f"OOD covariance inverse not found: {cov_inv_path}"
        )

    class_means = np.load(class_means_path)
    cov_inv = np.load(cov_inv_path)
    return class_means, cov_inv


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def _preprocess(image_path):
    """
    Convert an image into a normalized NCHW float32 tensor.
    """

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (224, 224),
            Image.BILINEAR,
        )

        array = (
            np.asarray(image)
            .astype(np.float32)
            / 255.0
        )

    array = (array - MEAN) / STD

    # HWC -> CHW
    array = array.transpose(2, 0, 1)

    # CHW -> NCHW
    array = np.expand_dims(
        array,
        axis=0,
    ).astype(np.float32)

    return array


def _preprocess_pil_image(image):
    """
    Preprocess a PIL image into a normalized NCHW tensor.
    Used by TTA inference.
    """

    image = image.convert("RGB")
    image = image.resize(
        (224, 224),
        Image.BILINEAR,
    )

    array = (
        np.asarray(image)
        .astype(np.float32)
        / 255.0
    )

    array = (array - MEAN) / STD

    array = array.transpose(2, 0, 1)

    array = np.expand_dims(
        array,
        axis=0,
    ).astype(np.float32)

    return array


# ============================================================
# SOFTMAX
# ============================================================

def _softmax(logits, temperature=1.0):
    """
    Numerically stable softmax with temperature scaling.
    """

    temperature = max(
        float(temperature),
        1e-6,
    )

    scaled = (
        logits - np.max(logits)
    ) / temperature

    exponentials = np.exp(scaled)

    return (
        exponentials
        / np.sum(exponentials)
    )


# ============================================================
# OUT-OF-DISTRIBUTION (OOD) DETECTION
# ============================================================

def _mahalanobis_distance(feature_vector, mean_vector, cov_inv):
    """
    Mahalanobis distance between a feature vector and a class mean,
    given the shared inverse covariance matrix.
    """
    diff = feature_vector - mean_vector
    return float(np.sqrt(diff @ cov_inv @ diff.T))


def compute_ood_distance(feature_vector, class_means, cov_inv):
    """
    Minimum Mahalanobis distance from the given feature vector to any of
    the six trained class means. Large values indicate the image's
    features fall far outside anything seen during training — e.g. it
    is not actually a dog skin photo at all.
    """
    distances = [
        _mahalanobis_distance(feature_vector, class_means[c], cov_inv)
        for c in range(class_means.shape[0])
    ]
    return min(distances)


# ============================================================
# DOG VALIDATION GATE
# ============================================================

def is_likely_dog(
    general_model,
    image_path,
    min_mass=DOG_CONFIDENCE_THRESHOLD,
    check_top_k=5,
):
    """
    Validate whether an uploaded image is likely to contain a dog.

    Validation rules:

    1. If the top-1 ImageNet prediction is a dog breed,
       accept the image.

    2. Otherwise, if a dog breed appears in the top-k predictions
       and cumulative dog probability reaches the threshold,
       accept the image.

    3. Otherwise, if cumulative probability across all dog
       classes reaches the threshold, accept the image.

    Returns:
        True  -> image passes dog validation
        False -> image fails dog validation
    """

    input_array = _preprocess(image_path)

    input_name = general_model.get_inputs()[0].name

    outputs = general_model.run(
        None,
        {input_name: input_array},
    )[0]

    logits = np.asarray(outputs).squeeze()

    probabilities = _softmax(
        logits,
        temperature=1.0,
    )

    if len(probabilities) < 269:
        raise ValueError(
            "General model output does not contain enough "
            "classes for the configured ImageNet dog mapping."
        )

    top_indices = np.argsort(
        probabilities
    )[::-1]

    top1_class = int(
        top_indices[0]
    )

    top_k_indices = top_indices[
        :check_top_k
    ]

    dog_probability_mass = float(
        np.sum(
            probabilities[
                list(DOG_CLASS_RANGE)
            ]
        )
    )

    has_dog_in_top_k = any(
        int(index) in DOG_CLASS_RANGE
        for index in top_k_indices
    )

    if top1_class in DOG_CLASS_RANGE:
        return True

    if (
        has_dog_in_top_k
        and dog_probability_mass >= min_mass
    ):
        return True

    if dog_probability_mass >= min_mass:
        return True

    return False


# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_image(
    session,
    image_path,
    class_means,
    cov_inv,
    use_tta=False,
    temperature=CALIBRATION_TEMPERATURE,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    ood_threshold=OOD_THRESHOLD,
):
    """
    Run disease-detection inference with calibrated confidence,
    confidence-based rejection, and out-of-distribution detection.

    The model has two ONNX outputs: 'logits' (classification logits)
    and 'features' (pooled penultimate-layer features used for OOD
    detection). TTA mode averages probabilities and features across
    augmented variants.

    Returns a dict always containing a "status" key:
        "not_recognized"     -> image's features are too far from
                                 anything seen in training (OOD)
        "unable_to_classify" -> recognized as plausible input, but
                                 confidence is below the threshold
        "possible_condition" -> confident, calibrated prediction
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    if not use_tta:

        input_array = _preprocess(image_path)

        logits, features = session.run(
            output_names,
            {input_name: input_array},
        )

        logits = np.asarray(logits).squeeze()
        features = np.asarray(features).squeeze()

        probabilities = _softmax(
            logits,
            temperature=temperature,
        )

    else:

        with Image.open(image_path) as original:

            image = original.convert("RGB")

            variants = [
                image,
                ImageOps.mirror(image),
                image.rotate(10, expand=False),
                image.rotate(-10, expand=False),
            ]

            all_probabilities = []
            all_features = []

            for variant in variants:

                input_array = _preprocess_pil_image(variant)

                logits, features = session.run(
                    output_names,
                    {input_name: input_array},
                )

                logits = np.asarray(logits).squeeze()
                features = np.asarray(features).squeeze()

                variant_probabilities = _softmax(
                    logits,
                    temperature=temperature,
                )

                all_probabilities.append(variant_probabilities)
                all_features.append(features)

            probabilities = np.mean(all_probabilities, axis=0)
            features = np.mean(all_features, axis=0)

    # --------------------------------------------------------
    # Validate model output
    # --------------------------------------------------------

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Disease model returned "
            f"{len(probabilities)} classes, but "
            f"{len(CLASS_NAMES)} class names are configured."
        )

    # --------------------------------------------------------
    # Out-of-distribution check — run BEFORE trusting the
    # classifier's output at all.
    # --------------------------------------------------------

    ood_distance = compute_ood_distance(features, class_means, cov_inv)

    if ood_distance > ood_threshold:
        return {
            "status": "not_recognized",
            "prediction": None,
            "confidence": None,
            "ood_distance": round(ood_distance, 2),
            "second_prediction": None,
            "second_confidence": None,
            "is_ambiguous": False,
            "all_probabilities": None,
            "message": (
                "This image doesn't appear to be a dog skin photo. "
                "Please upload a clear image of the affected area."
            ),
        }

    # --------------------------------------------------------
    # Top-2 predictions
    # --------------------------------------------------------

    top2_indices = np.argsort(probabilities)[::-1][:2]

    top_prediction_index = int(top2_indices[0])
    second_prediction_index = int(top2_indices[1])

    top_prediction = CLASS_NAMES[top_prediction_index]
    top_confidence = float(probabilities[top_prediction_index])

    second_prediction = CLASS_NAMES[second_prediction_index]
    second_confidence = float(probabilities[second_prediction_index])

    is_ambiguous = (top_confidence - second_confidence) < 0.15

    all_probabilities = {
        CLASS_NAMES[index]: float(probabilities[index])
        for index in range(len(CLASS_NAMES))
    }

    # --------------------------------------------------------
    # Confidence-based rejection
    # --------------------------------------------------------

    if top_confidence < confidence_threshold:
        return {
            "status": "unable_to_classify",
            "prediction": None,
            "confidence": top_confidence,
            "ood_distance": round(ood_distance, 2),
            "second_prediction": second_prediction if is_ambiguous else None,
            "second_confidence": second_confidence if is_ambiguous else None,
            "is_ambiguous": is_ambiguous,
            "all_probabilities": all_probabilities,
            "message": (
                "Unable to classify with sufficient confidence. "
                "Please consult a veterinarian for an accurate diagnosis."
            ),
        }

    return {
        "status": "possible_condition",
        "prediction": top_prediction,
        "confidence": top_confidence,
        "ood_distance": round(ood_distance, 2),
        "second_prediction": second_prediction if is_ambiguous else None,
        "second_confidence": second_confidence if is_ambiguous else None,
        "is_ambiguous": is_ambiguous,
        "all_probabilities": all_probabilities,
        "message": (
            f"Possible condition: {top_prediction}. This is a preliminary "
            f"AI screening result, not a diagnosis — please consult a "
            f"veterinarian to confirm."
        ),
    }