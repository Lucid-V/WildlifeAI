
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BASE = Path(__file__).parent
DATASET = BASE / "dataset"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Prepare images
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder(
    DATASET / "train",
    transform=train_transform
)

val_data = datasets.ImageFolder(
    DATASET / "val",
    transform=val_transform
)

if train_data.classes != val_data.classes:
    raise ValueError("Training and validation classes do not match.")

if len(train_data) == 0 or len(val_data) == 0:
    raise ValueError("Add images to both train and val folders.")

print("Classes:", train_data.classes)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))

train_loader = DataLoader(
    train_data, batch_size=4, shuffle=True
)

val_loader = DataLoader(
    val_data, batch_size=4, shuffle=False
)

# Load pretrained ResNet18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace the final layer for our 5 classes
model.fc = nn.Linear(
    model.fc.in_features,
    len(train_data.classes)
)

model = model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Validation accuracy: {accuracy:.1f}%"
    )

# Save model and class names
save_path = MODEL_DIR / "wildlife_classifier.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": train_data.classes
}, save_path)

print("\nTraining complete!")
print("Model saved to:", save_path)
print("NOTE: One image per class is only a pipeline test.")
print("Validation accuracy is not a reliable performance estimate.")
