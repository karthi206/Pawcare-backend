import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from cnn_model import load_model, predict_image, CLASS_NAMES, load_general_model, is_likely_dog

# Production model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pawcare_model.onnx')
GENERAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'general_imagenet_model.onnx')

# Data path (search common relative paths)
DATA_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'),
    os.path.join(os.path.dirname(__file__), '..', 'data'),
    os.path.join(os.path.dirname(__file__), 'data'),
]
DATA_DIR = next((d for d in DATA_DIR_CANDIDATES if os.path.exists(d)), DATA_DIR_CANDIDATES[0])


def compute_ece(confidences, predictions, ground_truths, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    Measures how closely predicted softmax confidence matches empirical accuracy across probability bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    print("\n--- Confidence Calibration Table (ECE Bins) ---")
    print(f"{'Bin Range':<15} | {'Count':<8} | {'Avg Conf':<10} | {'Avg Acc':<10} | {'Gap':<10}")
    print("-" * 65)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = [
            (c, p == y)
            for c, p, y in zip(confidences, predictions, ground_truths)
            if bin_lower < c <= bin_upper or (i == 0 and c == bin_lower)
        ]

        if len(in_bin) > 0:
            bin_conf = np.mean([item[0] for item in in_bin])
            bin_acc = np.mean([item[1] for item in in_bin])
            bin_weight = len(in_bin) / total_samples
            gap = abs(bin_acc - bin_conf)
            ece += bin_weight * gap

            print(f"[{bin_lower:.2f}, {bin_upper:.2f}]     | {len(in_bin):<8} | {bin_conf:<10.4f} | {bin_acc:<10.4f} | {gap:<10.4f}")

    print("-" * 65)
    return ece


def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: ONNX model not found at {MODEL_PATH}")
        return

    print(f"Loading ONNX production model: {MODEL_PATH}")
    session = load_model(MODEL_PATH)

    general_model = None
    if os.path.exists(GENERAL_MODEL_PATH):
        print(f"Loading General ImageNet model for dog gate evaluation: {GENERAL_MODEL_PATH}")
        general_model = load_general_model(GENERAL_MODEL_PATH)

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        return

    print(f"Evaluating dataset from: {DATA_DIR}\n")

    y_true = []
    y_pred = []
    confidences = []
    dog_gate_results = []

    for label in os.listdir(DATA_DIR):
        label_folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_folder):
            continue

        if label not in CLASS_NAMES:
            print(f"Note: Folder '{label}' is not in CLASS_NAMES, skipping.")
            continue

        image_files = [f for f in os.listdir(label_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        print(f"Evaluating class '{label}' ({len(image_files)} images)...")

        for img_name in image_files:
            img_path = os.path.join(label_folder, img_name)
            try:
                res = predict_image(session, img_path, use_tta=False)
                pred = res["prediction"]
                conf = res["confidence"]

                y_true.append(label)
                y_pred.append(pred)
                confidences.append(conf)

                if general_model is not None:
                    dog_gate_results.append(is_likely_dog(general_model, img_path))

            except Exception as e:
                print(f"  Error processing {img_path}: {e}")

    total = len(y_true)
    if total == 0:
        print("No valid test images found.")
        return

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total

    print(f"\n==========================================")
    print(f"🎯 EVALUATION REPORT (ONNX Production Model)")
    print(f"==========================================")
    print(f"Total Evaluated Images: {total}")
    print(f"Overall Accuracy:       {accuracy * 100:.2f}% ({correct}/{total})")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    header_title = "True \\ Pred"
    print(f"{header_title:<20} " + " ".join([f"{c[:6]:>7}" for c in CLASS_NAMES]))
    for idx, row in enumerate(cm):
        print(f"{CLASS_NAMES[idx]:<20} " + " ".join([f"{val:>7}" for val in row]))

    # Compute Calibration (ECE)
    ece = compute_ece(confidences, y_pred, y_true)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f} ({ece * 100:.2f}%)")

    if general_model is not None and dog_gate_results:
        dog_pass_rate = sum(1 for d in dog_gate_results if d) / len(dog_gate_results)
        print(f"\nDog Detection Gate Pass Rate on Dog Disease Data: {dog_pass_rate * 100:.2f}% ({sum(1 for d in dog_gate_results if d)}/{len(dog_gate_results)})")


if __name__ == '__main__':
    evaluate_model()