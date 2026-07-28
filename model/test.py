import os
from collections import defaultdict

DATA_DIR = '../data'
filename_locations = defaultdict(list)

for label in os.listdir(DATA_DIR):
    label_folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_folder):
        continue
    for img_name in os.listdir(label_folder):
        filename_locations[img_name].append(label)

duplicates = {name: labels for name, labels in filename_locations.items() if len(labels) > 1}
print(f"Found {len(duplicates)} filenames appearing in multiple class folders:")
for name, labels in list(duplicates.items())[:20]:
    print(f"  {name}: {labels}")