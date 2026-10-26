from pathlib import Path

import numpy as np
import torch
from PIL import Image

OPENEARTHMAP_CLASS_MAPPING = {
    0: (
        "Background",
        (0, 0, 0),
    ),  # OEM classes start from 1 on, so we add this for indexing
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

WHUOPTSAR_CLASS_MAPPING = {
    0: ("Background", (0, 0, 0)),
    10: ("Farmland", (204, 102, 1)),
    20: ("City", (254, 0, 0)),
    30: ("Village", (255, 255, 1)),
    40: ("Water", (0, 0, 254)),
    50: ("Forest", (85, 166, 1)),
    60: ("Road", (93, 255, 255)),
    70: ("Others", (152, 102, 153)),
}

WHUOPTSAR_COLOR_MAPPING = np.array(
    [
        [0, 0, 0],
        [204, 102, 1],
        [254, 0, 0],
        [255, 255, 1],
        [0, 0, 254],
        [85, 166, 1],
        [93, 255, 255],
        [152, 102, 153],
    ],
    dtype=np.uint8,
)


def get_num_classes(dataset_name: str) -> int:
    if dataset_name == "whu":
        mapping = WHUOPTSAR_CLASS_MAPPING
    elif dataset_name == "oem":
        mapping = OPENEARTHMAP_CLASS_MAPPING
    else:
        raise ValueError(f"Dataset '{dataset_name}' unsupported.")
    return len(mapping)


def draw_segment_image(
    label_file: str | Path | torch.Tensor | np.ndarray,
    dataset_name: str,
    show: bool = False,
    to_pil_image: bool = False,
) -> np.ndarray | Image.Image:
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
        case "whu" | "whuoptsar" | "whu-opt-sar":
            return WHUOPTSAR_COLOR_MAPPING
        case _:
            raise NotImplementedError(
                f"'{dataset_name}' has no implemented color scheme."
            )


def load_label_image(
    labels: str | Path | np.ndarray | torch.Tensor,
) -> np.ndarray:
    match labels:
        case np.ndarray():
            return labels
        case torch.Tensor():
            return labels.detach().cpu().numpy()
        case str() | Path():
            label_img = Image.open(labels)
            return np.asarray(label_img)
        case _:
            raise ValueError(f"'{type(labels)}' not supported.")


def map_classes_to_labels(dataset_name: str):
    match dataset_name.lower():
        case "oem" | "openearthmap" | "open-earth-map":
            return [
                class_tuple[0] for class_tuple in OPENEARTHMAP_CLASS_MAPPING.values()
            ]
        case "whu" | "whuoptsar" | "whu-opt-sar":
            return [class_tuple[0] for class_tuple in WHUOPTSAR_CLASS_MAPPING.values()]
        case _:
            raise ValueError(f"'{dataset_name}' not supported.")
