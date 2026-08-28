import os

DATA_DIR = '../data'

for label in os.listdir(DATA_DIR):
    label_folder = os.path.join(DATA_DIR, label)
    if os.path.isdir(label_folder):
        count = len(os.listdir(label_folder))
        print(f"{label}: {count} images")