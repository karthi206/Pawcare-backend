import sys, os
sys.path.append(os.path.dirname(__file__))
from cnn_model import load_model, predict_image

model = load_model('pawcare_cnn_model_v3.pth')

DATA_DIR = '../data'
TEST_FILENAMES_PATH = 'test_set_filenames(1).txt'

# Load the exact set of filenames that were genuinely held out during Colab training
with open(TEST_FILENAMES_PATH, 'r') as f:
    test_filenames = set(line.strip() for line in f)

print(f"Loaded {len(test_filenames)} known held-out test filenames.\n")

correct = 0
total = 0

for label in os.listdir(DATA_DIR):
    label_folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_folder):
        continue

    for img_name in os.listdir(label_folder):
        # Only evaluate images that were genuinely part of the test split
        if img_name not in test_filenames:
            continue

        img_path = os.path.join(label_folder, img_name)
        result = predict_image(model, img_path, use_tta=False)
        prediction = result["prediction"]
        confidence = result["confidence"]

        total += 1
        if prediction == label:
            correct += 1
        else:
            print(f"WRONG: actual={label}, predicted={prediction} ({confidence:.2f}), file={img_name}")

print(f"\nAccuracy on genuine held-out test set: {correct}/{total} = {100*correct/total:.1f}%")