import math
import os
import random

import numpy as np
from PIL import Image

# Configuration
DATASET_DIR = "path/to/whu-opt-sar/"  # Path to dataset root directory
FOLDERS = ["lbl", "optical", "sar"]
NUM_CROPS = 50
CROP_SIZE = (512, 512)  # (Width, Height) of output crops
SEED = 42

TRAIN_SPLIT_FILE = os.path.join(DATASET_DIR, "train.txt")
VAL_SPLIT_FILE = os.path.join(DATASET_DIR, "val.txt")

OUTPUT_TRAIN_SPLIT = os.path.join(DATASET_DIR, "train_crops.txt")
OUTPUT_VAL_SPLIT = os.path.join(DATASET_DIR, "val_crops.txt")


def process_split(split_file_path, output_split_path, crop_size, num_crops, seed):
    random.seed(seed)
    np.random.seed(seed)

    crop_w, crop_h = crop_size

    # Required radius around center to prevent black boundary artifacts during rotation
    required_margin = math.ceil(math.sqrt(crop_w**2 + crop_h**2) / 2)

    with open(split_file_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    new_split_filenames = []

    for filename in filenames:
        base_name, ext = os.path.splitext(filename)

        # Load input images from all modal folders
        images = {}
        for folder in FOLDERS:
            img_path = os.path.join(DATASET_DIR, folder, filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image: {img_path}")
            images[folder] = Image.open(img_path)

        # Verify equal image dimensions across folders
        img_w, img_h = images[FOLDERS[0]].size
        for folder in FOLDERS[1:]:
            if images[folder].size != (img_w, img_h):
                raise ValueError(
                    f"Dimension mismatch in {filename} between {FOLDERS[0]} and {folder}"
                )

        # Ensure original image is large enough for the rotated crop
        if img_w < 2 * required_margin or img_h < 2 * required_margin:
            raise ValueError(
                f"Image {filename} size ({img_w}x{img_h}) is too small for crop size {crop_size} "
                f"with rotation margin requirement ({2 * required_margin}x{2 * required_margin})."
            )

        # Generate synchronized crops
        for i in range(1, num_crops + 1):
            # Sample crop parameters
            center_x = random.randint(required_margin, img_w - required_margin)
            center_y = random.randint(required_margin, img_h - required_margin)
            angle = random.uniform(0.0, 360.0)

            crop_filename = f"{base_name}_{i:02d}{ext}"

            # Calculate crop bounding box coordinates centered at (center_x, center_y)
            left = center_x - crop_w // 2
            top = center_y - crop_h // 2
            right = left + crop_w
            bottom = top + crop_h

            # Apply identical transformation to each folder
            for folder, img in images.items():
                # Nearest neighbor for segmentation masks (lbl), Bilinear for imagery (opt, sar)
                resample_mode = (
                    Image.Resampling.NEAREST
                    if folder == "lbl"
                    else Image.Resampling.BILINEAR
                )

                # Rotate around chosen center point and crop patch
                rotated_img = img.rotate(
                    angle, resample=resample_mode, center=(center_x, center_y)
                )
                crop_patch = rotated_img.crop((left, top, right, bottom))

                out_folder = os.path.join(DATASET_DIR, folder)
                os.makedirs(out_folder, exist_ok=True)
                crop_patch.save(os.path.join(out_folder, crop_filename))

            new_split_filenames.append(crop_filename)

    # Write updated split text file
    with open(output_split_path, "w") as f:
        f.writelines(f"{fname}\n" for fname in new_split_filenames)


if __name__ == "__main__":
    print("Processing train split...")
    process_split(TRAIN_SPLIT_FILE, OUTPUT_TRAIN_SPLIT, CROP_SIZE, NUM_CROPS, seed=SEED)

    print("Processing val split...")
    # Using offset seed to maintain independence between train and val sampling
    process_split(
        VAL_SPLIT_FILE, OUTPUT_VAL_SPLIT, CROP_SIZE, NUM_CROPS, seed=SEED + 1000
    )

    print("Crop dataset generation complete.")
