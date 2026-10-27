
import json
from pathlib import Path

import torch
from PytorchWildlife.models import detection
from PytorchWildlife import utils

BASE = Path(__file__).parent
IMAGE_DIR = BASE / "images"
RESULT_DIR = BASE / "results"

RESULT_DIR.mkdir(exist_ok=True)

images = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    images.extend(IMAGE_DIR.glob(ext))

if not images:
    print("No images found in:", IMAGE_DIR)
    print("Add a wildlife JPG or PNG image first.")
    raise SystemExit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Loading MegaDetector V6...")

model = detection.MegaDetectorV6(
    device=device,
    pretrained=True,
    version="MDV6-yolov10-e"
)

image_path = images[0]
print("Analyzing:", image_path.name)

result = model.single_image_detection(str(image_path))

json_path = RESULT_DIR / "detection.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, default=str)

utils.save_detection_images(
    result,
    str(RESULT_DIR),
    overwrite=True
)

print("Detection finished!")
print("JSON:", json_path)
print("Images and results:", RESULT_DIR)
