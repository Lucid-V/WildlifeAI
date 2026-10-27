
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from PytorchWildlife.models import detection

BASE = Path(__file__).parent
IMAGE_DIR = BASE / "images"
RESULT_DIR = BASE / "results"
MODEL_PATH = BASE / "models" / "wildlife_classifier.pth"

RESULT_DIR.mkdir(exist_ok=True)

# Choose the image

# Automatically select the newest image in the images folder
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
]

if not image_files:
    raise FileNotFoundError(
        f"No images found in {IMAGE_DIR}. Add a wildlife photo first."
    )

image_path = max(image_files, key=lambda p: p.stat().st_mtime)

print("Automatically selected image:", image_path.name)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load the trained species classifier
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

classifier = models.resnet18(weights=None)
classifier.fc = torch.nn.Linear(
    classifier.fc.in_features, len(classes)
)
classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load MegaDetector
print("Loading MegaDetector...")
detector = detection.MegaDetectorV6(
    device=device,
    pretrained=True,
    version="MDV6-yolov10-e"
)

# Detect animals in the original image
print("Detecting animals in:", image_path.name)
detections = detector.single_image_detection(str(image_path))

original = Image.open(image_path).convert("RGB")
width, height = original.size

boxes = detections["detections"].xyxy
detector_confidences = detections["detections"].confidence
detector_classes = detections["detections"].class_id

reports = []

for i, box in enumerate(boxes):
    detector_class = int(detector_classes[i])

    # MegaDetector class 0 is animal.
    # Skip people and vehicles.
    if detector_class != 0:
        continue

    x1, y1, x2, y2 = [int(v) for v in box]

    # Keep crop coordinates inside the image
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    crop = original.crop((x1, y1, x2, y2))
    crop_tensor = transform(crop).unsqueeze(0).to(device)

    # Predict species from the animal crop
    with torch.no_grad():
        output = classifier(crop_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        index = int(probabilities.argmax())

    reports.append({
        "animal_number": i + 1,
        "detector_confidence": float(detector_confidences[i]),
        "predicted_species": classes[index],
        "species_confidence": float(probabilities[index]),
        "bounding_box_xyxy": [x1, y1, x2, y2]
    })

result = {
    "image": image_path.name,
    "animal_count": len(reports),
    "detections": reports,
    "warning": (
        "Experimental classifier trained on very few images. "
        "Species predictions are not reliable yet."
    )
}

# Save a fresh copy of the selected image
# with the detected animal bounding boxes
from PIL import ImageDraw

annotated = original.copy()
draw = ImageDraw.Draw(annotated)

for item in reports:
    x1, y1, x2, y2 = item["bounding_box_xyxy"]
    label = (
        f'{item["predicted_species"]} '
        f'{item["species_confidence"]:.2f}'
    )

    draw.rectangle((x1, y1, x2, y2), outline="lime", width=4)
    draw.text((x1, max(0, y1 - 20)), label, fill="lime")

annotated_path = RESULT_DIR / f"{image_path.stem}_annotated.jpg"
annotated.save(annotated_path, quality=95)

print("Annotated image saved to:", annotated_path)

output_path = RESULT_DIR / f"{image_path.stem}_report.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("\nWildlife analysis complete!")
print(json.dumps(result, indent=2))
print("\nSaved report:", output_path)
