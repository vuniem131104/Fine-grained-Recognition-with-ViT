import albumentations as A
import cv2
from pathlib import Path

DATA_DIR = Path("/home/lehoangvu/AIDE_Project/data/CUB_200_2011")
IMAGES_DIR = DATA_DIR / "images"
VERSIONS_PER_IMAGE = 3

aug_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
    ], p=0.5),
    A.Affine(
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        scale=(0.9, 1.1),
        rotate=(-15, 15),
        p=0.5,
    ),
])

# Build mapping and identify training images
with open(DATA_DIR / "images.txt") as f:
    id_to_path = {img_id: rel_path for img_id, rel_path in (line.strip().split() for line in f)}

with open(DATA_DIR / "train_test_split.txt") as f:
    train_paths = {
        IMAGES_DIR / id_to_path[img_id]: img_id
        for img_id, is_train in (line.strip().split() for line in f)
        if is_train == "1"
    }

print(f"Found {len(train_paths)} training images to augment.")

next_id = max(int(k) for k in id_to_path) + 1
new_images_lines = []
new_split_lines = []

for idx, (img_path, _) in enumerate(sorted(train_paths.items()), 1):
    if "_aug" in img_path.stem:
        continue

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  WARNING: could not read {img_path.name}, skipping")
        continue

    stem = img_path.stem
    ext = img_path.suffix
    rel_folder = img_path.parent.relative_to(IMAGES_DIR)

    for i in range(1, VERSIONS_PER_IMAGE + 1):
        augmented = aug_transform(image=image)["image"]
        aug_filename = f"{stem}_aug{i}{ext}"
        out_path = img_path.parent / aug_filename
        cv2.imwrite(str(out_path), augmented)

        rel_path = str(rel_folder / aug_filename)
        new_images_lines.append(f"{next_id} {rel_path}")
        new_split_lines.append(f"{next_id} 1")
        next_id += 1

    if idx % 500 == 0:
        print(f"  Processed {idx}/{len(train_paths)}")

# Append new entries to images.txt and train_test_split.txt
with open(DATA_DIR / "images.txt", "a") as f:
    f.write("\n".join(new_images_lines) + "\n")

with open(DATA_DIR / "train_test_split.txt", "a") as f:
    f.write("\n".join(new_split_lines) + "\n")

print(f"Done. Added {len(new_images_lines)} augmented entries to images.txt and train_test_split.txt.")
