from pathlib import Path

import numpy as np
import torch
from PIL import Image

OPENEARTHMAP_CLASS_MAPPING = {
    1: ("Bareland", (128, 0, 0)),
    2: ("Rangeland", (0, 255, 36)),
    3: ("Developed space", (148, 148, 148)),
    4: ("Road", (255, 255, 255)),
    5: ("Tree", (34, 97, 38)),
    6: ("Water", (0, 69, 255)),
    7: ("Agriculture land", (75, 181, 73)),
    8: ("Building", (222, 31, 7)),
}

OPENEARTHMAP_COLOR_MAPPING = np.array(
    [
        [0, 0, 0],  # OEM classes start from 1 on, so we add this for indexing
        [128, 0, 0],
        [0, 255, 36],
        [148, 148, 148],
        [255, 255, 255],
        [34, 97, 38],
        [0, 69, 255],
        [75, 181, 73],
        [222, 31, 7],
    ],
    dtype=np.uint8,
)


def draw_segment_image(
    label_file: str | Path | torch.Tensor | np.ndarray,
    dataset_name: str,
    show: bool = False,
    to_pil_image: bool = False,
):
    """Makes a segment image from label image with the dataset-specific color scheme and shows it optionally.

    Args:
        label_file (str | Path | torch.Tensor | np.ndarray): label image to draw segment from
        dataset_name (str): dataset of input
        show (bool, optional): whether to show the segment image directly. Defaults to False.
        to_pil_image (bool, optional): whether to convert to PIL image. Defaults to False.

    Returns:
        Image: RGB segment PIL-Image
    """
    label_img = load_label_image(label_file)
    color_mapping = get_dataset_color_scheme(dataset_name)
    img_segment = color_mapping[label_img]
    if to_pil_image:
        img_segment = Image.fromarray(img_segment)
    if show:
        img_segment.show()

    return img_segment


def save_image(img: Image.Image, fname: str):
    img.save(fname)


def get_dataset_color_scheme(dataset_name: str) -> np.ndarray:
    match dataset_name.lower():
        case "oem" | "openearthmap" | "open-earth-map":
            return OPENEARTHMAP_COLOR_MAPPING
        case _:
            raise NotImplementedError(
                f"'{dataset_name}' has no implemented color scheme."
            )


def load_label_image(
    label_file_path: str | Path | np.ndarray,
) -> np.ndarray:
    match label_file_path:
        case np.ndarray():
            return label_file_path
        case str() | Path():
            label_img = Image.open(label_file_path)
            return np.asarray(label_img)
        case _:
            raise ValueError(f"'{type(label_file_path)}' not supported.")
