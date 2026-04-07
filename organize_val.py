"""Move flat val images into class subfolders based on filename prefix."""
from pathlib import Path
import shutil

val_dir = Path("data/processed/val")
classes = ["rock", "paper", "scissors"]

for cls in classes:
    (val_dir / cls).mkdir(exist_ok=True)

moved = 0
skipped = 0
for img in val_dir.iterdir():
    if not img.is_file():
        continue
    matched = False
    for cls in classes:
        if img.name.lower().startswith(cls):
            shutil.move(str(img), str(val_dir / cls / img.name))
            moved += 1
            matched = True
            break
    if not matched:
        print(f"Skipped (no class match): {img.name}")
        skipped += 1

print(f"Done. Moved: {moved}, Skipped: {skipped}")
