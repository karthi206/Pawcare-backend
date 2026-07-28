import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image

CLASS_NAMES = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']

DOG_CLASS_RANGE = range(151, 269)  # ImageNet class indices 151-268 are dog breeds
DOG_CONFIDENCE_THRESHOLD = 0.05    # sum of probability mass in dog classes must exceed this

def load_model(weights_path):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_general_model():
    """Loads the original, unmodified ImageNet classifier - used only to check 'is this a dog?'"""
    general_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    general_model.eval()
    return general_model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def is_likely_dog(general_model, image_path):
    """Returns True if the image is likely to contain a dog, based on ImageNet's dog breed classes."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = general_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    dog_probability_mass = sum(probabilities[i].item() for i in DOG_CLASS_RANGE)
    print(f"DEBUG: dog_probability_mass = {dog_probability_mass:.4f}")
    return dog_probability_mass >= DOG_CONFIDENCE_THRESHOLD


def predict_image(model, image_path, use_tta=True):
    image = Image.open(image_path).convert('RGB')

    if not use_tta:
        # Original single-pass prediction (kept available for comparison/debugging)
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    else:
        # Test-time augmentation: run several slightly-varied versions of the
        # same image through the model, then average their predictions together.
        variants = [
            image,                    # original
            TF.hflip(image),          # horizontally flipped
            TF.rotate(image, 10),     # slightly rotated one way
            TF.rotate(image, -10),    # slightly rotated the other way
        ]

        all_probabilities = []
        with torch.no_grad():
            for variant in variants:
                variant_tensor = transform(variant).unsqueeze(0)
                outputs = model(variant_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                all_probabilities.append(probs)

        probabilities = torch.stack(all_probabilities).mean(dim=0)

    # Get the top 2 predictions, sorted by confidence
    top2_confidences, top2_indices = torch.topk(probabilities, 2)

    top_prediction = CLASS_NAMES[top2_indices[0].item()]
    top_confidence = top2_confidences[0].item()

    second_prediction = CLASS_NAMES[top2_indices[1].item()]
    second_confidence = top2_confidences[1].item()

    # Flag as "ambiguous" if the top two predictions are close together
    # (e.g. 45% vs 40% - genuinely unclear - vs 90% vs 5% - clearly confident)
    is_ambiguous = (top_confidence - second_confidence) < 0.15

    return {
        "prediction": top_prediction,
        "confidence": top_confidence,
        "second_prediction": second_prediction if is_ambiguous else None,
        "second_confidence": second_confidence if is_ambiguous else None,
        "is_ambiguous": is_ambiguous,
    }