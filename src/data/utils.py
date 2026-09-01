import os
from pathlib import Path

import numpy as np


def map_data_dir_to_dataset(data_dir: str) -> str:
    dataset_dir = data_dir.split("/")[-1].lower()
    if dataset_dir == "whu-opt-sar":
        return "whu"
    elif dataset_dir.startswith("openearthmap"):
        return "oem"
    raise ValueError(f"Dataset '{dataset_dir}' not supported.")


def in_channels_from_data_type(data_type: str) -> int:
    match data_type.lower():
        case "rgb":
            return 3
        case "rgb+nir":
            return 4
        case "sar":
            return 1
        case _:
            raise ValueError(f"data_type '{data_type.lower()}' is unsupported")


def create_OEM_SAR_splits(data_dir, seed=42):
    """Create seeded random train/val split (95/5) for OEM-SAR dataset."""
    images_dir = os.path.join(data_dir, "train", "rgb_images")

    all_files = list(Path(images_dir).glob("*.tif"))
    if not all_files:
        raise FileNotFoundError(f"No files found in {images_dir}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_files))

    val_size = max(1, int(0.1 * len(indices)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    with open(os.path.join(data_dir, "train.txt"), "w") as f:
        f.writelines(all_files[idx].name + "\n" for idx in train_indices)

    with open(os.path.join(data_dir, "val.txt"), "w") as f:
        f.writelines(all_files[idx].name + "\n" for idx in val_indices)

    print("Finished creating splits!")
