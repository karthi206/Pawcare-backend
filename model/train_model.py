import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from features import extract_features
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DATA_DIR = '../data'

X = []  # feature vectors
y = []  # labels

# Step 1: Loop through each disease folder
for label in os.listdir(DATA_DIR):
    label_folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_folder):
        continue

    print(f"Processing class: {label}")

    for filename in os.listdir(label_folder):
        filepath = os.path.join(label_folder, filename)
        try:
            features = extract_features(filepath)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}, Feature vector size: {X.shape[1]}")

# Step 2: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 5: Save the trained model
joblib.dump(model, 'trained_model.pkl')
print("Model saved as trained_model.pkl")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved as confusion_matrix.png")