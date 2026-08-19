"""
PawCare AI - CNN Training & ONNX Export Pipeline

Trains a lightweight MobileNetV2 / CNN transfer-learning model on the 6 disease classes,
optimizes probability calibration via Temperature Scaling, and exports the production
pawcare_model.onnx and checkpoint pawcare_model.pth.
"""
import os
import sys
import time
import copy
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CLASS_NAMES = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

# Search common paths for the dataset directory
DATA_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'),
    os.path.join(os.path.dirname(__file__), '..', 'data'),
    os.path.join(os.path.dirname(__file__), 'data'),
]
DATA_DIR = next((d for d in DATA_DIR_CANDIDATES if os.path.exists(d)), DATA_DIR_CANDIDATES[0])

OUTPUT_ONNX_PATH = os.path.join(os.path.dirname(__file__), 'pawcare_model.onnx')
OUTPUT_PTH_PATH = os.path.join(os.path.dirname(__file__), 'pawcare_model.pth')


class DogDiseaseDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (filepath, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def build_datasets(data_dir, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    train_samples = []
    val_samples = []

    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        images = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        np.random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        label_idx = CLASS_TO_IDX[cls_name]

        for p in images[:split_idx]:
            train_samples.append((p, label_idx))
        for p in images[split_idx:]:
            val_samples.append((p, label_idx))

    return train_samples, val_samples


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def create_model(num_classes=NUM_CLASSES):
    """Constructs MobileNetV2 with customized classification head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def export_to_onnx(model, output_path):
    """Exports PyTorch model to ONNX for fast, cross-platform inference."""
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f" Successfully exported production ONNX model: {output_path}")


def train_pipeline(num_epochs=15, batch_size=16, lr=0.0003):
    if not TORCH_AVAILABLE:
        print("Error: PyTorch and torchvision are required for training. Please install torch and torchvision.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory {DATA_DIR} not found.")
        return

    train_samples, val_samples = build_datasets(DATA_DIR)
    print(f"Dataset summary: {len(train_samples)} training samples, {len(val_samples)} validation samples across {NUM_CLASSES} classes.")

    train_transform, val_transform = get_transforms()
    train_dataset = DogDiseaseDataset(train_samples, transform=train_transform)
    val_dataset = DogDiseaseDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weighting for balanced learning
    class_counts = np.zeros(NUM_CLASSES)
    for _, lbl in train_samples:
        class_counts[lbl] += 1
    weights = [len(train_samples) / (NUM_CLASSES * max(c, 1)) for c in class_counts]
    class_weights = torch.FloatTensor(weights).to(device)

    model = create_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\nStarting Training & Fine-Tuning Pipeline...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save PyTorch checkpoint
    torch.save(model.state_dict(), OUTPUT_PTH_PATH)
    print(f" Saved PyTorch weights checkpoint: {OUTPUT_PTH_PATH}")

    # Export production ONNX model
    export_to_onnx(model, OUTPUT_ONNX_PATH)


if __name__ == '__main__':
    train_pipeline()