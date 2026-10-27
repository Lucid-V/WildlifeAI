
from pathlib import Path
import json
import sys

import torch
from torchvision import models, transforms
from PIL import Image

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "wildlife_classifier.pth"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python predict.py images\\boar.jpg")
    raise SystemExit(1)

image_path = Path(sys.argv[1])

if not image_path.is_absolute():
    image_path = BASE / image_path

if not image_path.exists():
    print("Image not found:", image_path)
    raise SystemExit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and its class names
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Prepare the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict the class
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    index = int(probabilities.argmax())

species = classes[index]
confidence = float(probabilities[index])

result = {
    "image": image_path.name,
    "predicted_species": species,
    "confidence": round(confidence, 4),
    "warning": "Experimental model; not validated for real-world identification."
}

output_path = RESULT_DIR / "prediction.json"
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

print("\nPrediction complete!")
print(json.dumps(result, indent=2))
print("Saved to:", output_path)
