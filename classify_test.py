
import json
from pathlib import Path

import torch
from PytorchWildlife.models import classification

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)

image_path = IMAGE_DIR / "boar.jpg"

if not image_path.exists():
    print("Image not found:", image_path)
    print("Put boar.jpg inside the images folder.")
    raise SystemExit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
print("Loading pretrained classifier...")

model = classification.AI4GSnapshotSerengeti(
    device=device
)

print("Classifying:", image_path.name)

result = model.single_image_classification(
    str(image_path)
)

output_path = RESULT_DIR / "classification_test.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2, default=str)

print("\nClassification finished!")
print("Raw prediction:")
print(json.dumps(result, indent=2, default=str))
print("\nSaved to:", output_path)
print("Reminder: this classifier is not validated for Indian wildlife.")
