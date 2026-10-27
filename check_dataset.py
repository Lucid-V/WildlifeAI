
from pathlib import Path

base = Path("dataset")
classes = ["wild_boar", "deer", "leopard", "elephant", "other"]
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "val"]:
    print(f"\n--- {split.upper()} ---")

    for animal in classes:
        folder = base / split / animal

        if folder.exists():
            count = sum(
                1 for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            )
        else:
            count = 0

        print(f"{animal}: {count} images")
