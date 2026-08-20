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

# Dog-gate threshold.
#
# 0.05 is the currently tested value that allows the project's
# current valid dog images to pass the gate.
#
# Keep this configurable so it can be tuned without changing code.
DOG_CONFIDENCE_THRESHOLD = float(
    os.environ.get("DOG_GATE_THRESHOLD", "0.05")
)

# Temperature used for disease-model probability calibration.
CALIBRATION_TEMPERATURE = float(
    os.environ.get("AI_CALIBRATION_TEMPERATURE", "1.2")
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

    # Verify the model's expected input name instead of assuming
    # it is always "input".
    input_name = general_model.get_inputs()[0].name

    outputs = general_model.run(
        None,
        {input_name: input_array},
    )[0]

    # Remove unnecessary batch dimensions safely.
    logits = np.asarray(outputs).squeeze()

    probabilities = _softmax(
        logits,
        temperature=1.0,
    )

    # Make sure the model output is large enough for the
    # configured ImageNet dog-class range.
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

    # Rule 1: top-1 dog breed.
    if top1_class in DOG_CLASS_RANGE:
        return True

    # Rule 2: dog appears in top-k and threshold is reached.
    if (
        has_dog_in_top_k
        and dog_probability_mass >= min_mass
    ):
        return True

    # Rule 3: cumulative dog probability reaches threshold.
    if dog_probability_mass >= min_mass:
        return True

    return False


# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_image(
    session,
    image_path,
    use_tta=False,
    temperature=CALIBRATION_TEMPERATURE,
):
    """
    Run disease-detection inference.

    Returns:
        prediction
        confidence
        second_prediction
        second_confidence
        ambiguity information
        all class probabilities
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Get the ONNX model's actual input name.
    input_name = session.get_inputs()[0].name

    if not use_tta:

        input_array = _preprocess(
            image_path
        )

        logits = session.run(
            None,
            {input_name: input_array},
        )[0]

        logits = np.asarray(
            logits
        ).squeeze()

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
                image.rotate(
                    10,
                    expand=False,
                ),
                image.rotate(
                    -10,
                    expand=False,
                ),
            ]

            all_probabilities = []

            for variant in variants:

                input_array = (
                    _preprocess_pil_image(
                        variant
                    )
                )

                logits = session.run(
                    None,
                    {
                        input_name: input_array
                    },
                )[0]

                logits = np.asarray(
                    logits
                ).squeeze()

                variant_probabilities = _softmax(
                    logits,
                    temperature=temperature,
                )

                all_probabilities.append(
                    variant_probabilities
                )

            probabilities = np.mean(
                all_probabilities,
                axis=0,
            )

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
    # Top-2 predictions
    # --------------------------------------------------------

    top2_indices = np.argsort(
        probabilities
    )[::-1][:2]

    top_prediction_index = int(
        top2_indices[0]
    )

    second_prediction_index = int(
        top2_indices[1]
    )

    top_prediction = CLASS_NAMES[
        top_prediction_index
    ]

    top_confidence = float(
        probabilities[
            top_prediction_index
        ]
    )

    second_prediction = CLASS_NAMES[
        second_prediction_index
    ]

    second_confidence = float(
        probabilities[
            second_prediction_index
        ]
    )

    # --------------------------------------------------------
    # Ambiguity detection
    # --------------------------------------------------------

    is_ambiguous = (
        top_confidence
        - second_confidence
    ) < 0.15

    return {
        "prediction": top_prediction,
        "confidence": top_confidence,

        "second_prediction": (
            second_prediction
            if is_ambiguous
            else None
        ),

        "second_confidence": (
            second_confidence
            if is_ambiguous
            else None
        ),

        "is_ambiguous": is_ambiguous,

        "all_probabilities": {
            CLASS_NAMES[index]: float(
                probabilities[index]
            )
            for index in range(
                len(CLASS_NAMES)
            )
        },
    }