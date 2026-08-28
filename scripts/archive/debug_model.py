"""
PawCare AI - Standalone Model Diagnostic Script

Run this directly in your backend's `model/` folder (or point the paths
below at it). This bypasses Flask entirely and tests the ONNX models
directly, so we can tell whether the problem is:

  (a) the model files themselves (wrong file, bad export, wrong class order)
  (b) the dog-gate model (rejecting real dogs incorrectly)
  (c) something in the Flask glue code (app.py / cnn_model.py wiring)

USAGE:
    python debug_model.py

You'll need to edit TEST_DIR below to point at your test/ folder (the
same one with the 433 held-out images from the Kaggle split).
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

# ============================================================
# EDIT THESE PATHS
# ============================================================
MODEL_DIR = "."  # folder containing the .onnx and .npy files
TEST_DIR = "../data/test/ringworm/*.jpg"  # folder containing test/<ClassName>/*.jpg subfolders
DISEASE_MODEL_FILE = "pawcare_mobilenetv2_with_features.onnx"
GENERAL_MODEL_FILE = "general_imagenet_model.onnx"

# ============================================================
# CONFIG (must match cnn_model.py exactly)
# ============================================================
CLASS_NAMES = [
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "demodicosis",
    "ringworm",
]
DOG_CLASS_RANGE = range(151, 269)
DOG_CONFIDENCE_THRESHOLD = 0.05
CALIBRATION_TEMPERATURE = 1.5525

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((224, 224), Image.BILINEAR)
        array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    array = array.transpose(2, 0, 1)
    array = np.expand_dims(array, axis=0).astype(np.float32)
    return array


def softmax(logits, temperature=1.0):
    temperature = max(float(temperature), 1e-6)
    scaled = (logits - np.max(logits)) / temperature
    exponentials = np.exp(scaled)
    return exponentials / np.sum(exponentials)


def test_disease_model():
    print("=" * 60)
    print("TEST 1: Disease model accuracy on real test-set images")
    print("=" * 60)

    model_path = os.path.join(MODEL_DIR, DISEASE_MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"  MODEL FILE NOT FOUND: {model_path}")
        return
    if not os.path.isdir(TEST_DIR):
        print(f"  TEST_DIR NOT FOUND: {TEST_DIR} (edit the path at the top of this script)")
        return

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print(f"  Model outputs: {output_names}")

    if len(output_names) != 2:
        print("  WARNING: expected 2 outputs (logits, features) - this looks like the OLD single-output model!")

    correct = 0
    total = 0
    per_class = {c: {"correct": 0, "total": 0} for c in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"  Skipping missing class folder: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                input_array = preprocess(fpath)
                outputs = session.run(output_names, {input_name: input_array})
                logits = np.asarray(outputs[0]).squeeze()
                probs = softmax(logits, temperature=CALIBRATION_TEMPERATURE)
                pred_idx = int(np.argmax(probs))
                pred_class = CLASS_NAMES[pred_idx]

                total += 1
                per_class[class_name]["total"] += 1
                if pred_class == class_name:
                    correct += 1
                    per_class[class_name]["correct"] += 1
            except Exception as e:
                print(f"  ERROR on {fpath}: {e}")

    if total == 0:
        print("  No test images found - check TEST_DIR path")
        return

    print(f"\n  Overall accuracy: {correct}/{total} = {100*correct/total:.2f}%")
    print("  (Compare this to the 96.77% found in Colab)")
    print("\n  Per-class breakdown:")
    for c in CLASS_NAMES:
        t = per_class[c]["total"]
        cor = per_class[c]["correct"]
        if t > 0:
            print(f"    {c:20s}: {cor}/{t} = {100*cor/t:.1f}%")
        else:
            print(f"    {c:20s}: (no test images found)")


def test_dog_gate(sample_dog_images):
    print("\n" + "=" * 60)
    print("TEST 2: Dog-gate model on known real dog images")
    print("=" * 60)

    model_path = os.path.join(MODEL_DIR, GENERAL_MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"  MODEL FILE NOT FOUND: {model_path}")
        return

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for fpath in sample_dog_images:
        if not os.path.exists(fpath):
            print(f"  (skipping missing file: {fpath})")
            continue
        input_array = preprocess(fpath)
        outputs = session.run(None, {input_name: input_array})[0]
        logits = np.asarray(outputs).squeeze()

        print(f"\n  File: {fpath}")
        print(f"  Output vector length: {len(logits)}")

        if len(logits) < 269:
            print("  *** PROBLEM: this model has fewer than 269 output classes.")
            print("  *** The DOG_CLASS_RANGE (151-269) assumes standard ImageNet-1000")
            print("  *** class ordering. This model does NOT match that assumption.")
            continue

        probs = softmax(logits, temperature=1.0)
        top5_idx = np.argsort(probs)[::-1][:5]
        dog_mass = float(np.sum(probs[list(DOG_CLASS_RANGE)]))

        print(f"  Top-5 class indices: {list(top5_idx)}")
        print(f"  Top-5 probabilities: {[round(float(probs[i]), 4) for i in top5_idx]}")
        print(f"  Cumulative dog-class probability mass (indices 151-269): {dog_mass:.4f}")
        print(f"  Threshold: {DOG_CONFIDENCE_THRESHOLD}")
        print(f"  Would PASS dog gate: {top5_idx[0] in DOG_CLASS_RANGE or dog_mass >= DOG_CONFIDENCE_THRESHOLD}")


if __name__ == "__main__":
    test_disease_model()

    # EDIT THIS: point at a few real dog photos you have locally,
    # ideally ones that got incorrectly rejected as "not a dog" in the app
    sample_dogs = [
        # "path/to/a/dog/photo/that/got/rejected.jpg",
    ]
    if sample_dogs:
        test_dog_gate(sample_dogs)
    else:
        print("\n(Skipping dog-gate test - add file paths to sample_dogs list at the bottom of this script)")