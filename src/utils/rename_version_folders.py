"""
Script to rename version_ folders in tensorboard logs based on hparams.yaml configuration.

Naming schemes:
- att-dynamic   -> att_<initial_planes>_<num_layers>_<K>_<lr>_<dataset>_<data_type>
- stat          -> stat_<initial_planes>_<num_layers>_<lr>_<dataset>_<data_type>
- wp-dynamic    -> wp_<initial_planes>_<num_layers>_<kernel_gen_hidden>_<lr>_<dataset>_<data_type>
- wm-dynamic    -> wm_<initial_planes>_<num_layers>_<kernel_gen_hidden>_<lr>_<dataset>_<data_type>
- lora-dynamic  -> lora_<initial_planes>_<num_layers>_<rank>_<reduction>_<lr>_<dataset>_<data_type>
- depth-dynamic -> depth_<initial_planes>_<num_layers>_<reduction>_<lr>_<dataset>_<data_type>
"""

import argparse
import shutil
import sys
from pathlib import Path

# Dataset mapping
DATASET_MAP = {
    "OpenEarthMap-SAR": "oem",
    "whu-opt-sar": "whu",
}


def extract_dataset_from_data_dir(data_dir: str) -> str | None:
    """Extract dataset name from data_dir path."""
    if not data_dir or not data_dir.strip():
        return None

    normalized = data_dir.strip().rstrip("/")

    for full_dataset_name, short_name in DATASET_MAP.items():
        if full_dataset_name in normalized:
            return short_name

    return None


def parse_lr(lr_val) -> str:
    """Parse learning rate and format as string."""
    if lr_val is None:
        return "null"
    try:
        lr_float = float(lr_val)
        if abs(lr_float) < 0.001:
            return f"{lr_float:.1e}"
        return f"{lr_float:g}"
    except (ValueError, TypeError):
        return str(lr_val)


def get_new_name(conv_type: str, hparams: dict) -> str | None:
    """Generate new folder name based on conv_type and hparams."""
    dataset = extract_dataset_from_data_dir(hparams.get("data_dir", ""))
    if not dataset:
        return None

    initial_planes = hparams.get("initial_planes", 0)
    num_layers = hparams.get("num_layers", 0)
    lr_str = parse_lr(hparams.get("lr"))
    data_type = str(hparams.get("data_type", "sar")).lower()
    dyn_placement = str(hparams.get("dyn_placement", "full")).lower()

    # Common base parts
    parts = [str(initial_planes), str(num_layers)]

    if conv_type in ("att-dynamic", "att"):
        k_val = hparams.get("K")
        if k_val is not None:
            parts.append(str(k_val))
        parts.extend([lr_str, dataset, data_type, dyn_placement])
        return f"att_{'_'.join(parts)}"

    elif conv_type in ("stat", "static"):
        parts.extend([lr_str, dataset, data_type, dyn_placement])
        return f"stat_{'_'.join(parts)}"

    elif conv_type in ("wp-dynamic", "wp"):
        kh_val = hparams.get("kernel_gen_hidden")
        if kh_val is not None:
            parts.append(str(kh_val))
        parts.extend([lr_str, dataset, data_type, dyn_placement])
        return f"wp_{'_'.join(parts)}"

    elif conv_type in ("wm-dynamic", "wm"):
        kh_val = hparams.get("kernel_gen_hidden")
        if kh_val is not None:
            parts.append(str(kh_val))
        parts.extend([lr_str, dataset, data_type, dyn_placement])
        return f"wm_{'_'.join(parts)}"

    elif conv_type in ("lora-dynamic", "lora"):
        rank = hparams.get("rank", "")
        reduction = hparams.get("reduction", "")
        parts.extend([str(rank), str(reduction), lr_str, dataset, data_type])
        return f"lora_{'_'.join(parts)}"

    elif conv_type in ("depth-dynamic", "depth"):
        reduction = hparams.get("reduction", "")
        parts.extend([str(reduction), lr_str, dataset, data_type, dyn_placement])
        return f"depth_{'_'.join(parts)}"

    else:
        print(f"  Unknown conv_type: {conv_type}")
        return None


def load_hparams(hparams_path: Path) -> dict:
    """Load hparams.yaml file."""
    try:
        import yaml

        with open(hparams_path, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("  Warning: PyYAML not installed. Using basic YAML parsing.")
        hparams = {}
        try:
            with open(hparams_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line and not line.startswith("#"):
                        key, _, value = line.partition(":")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        try:
                            if "." in value or "e" in value.lower():
                                hparams[key] = float(value)
                            elif value.lower() == "true":
                                hparams[key] = True
                            elif value.lower() == "false":
                                hparams[key] = False
                            elif value.lower() == "null":
                                hparams[key] = None
                            else:
                                hparams[key] = int(value) if value.isdigit() else value
                        except ValueError:
                            hparams[key] = value
            return hparams
        except Exception as e:
            print(f"  Error reading hparams.yaml: {e}")
            return {}
    except Exception as e:
        print(f"  Error loading hparams from {hparams_path}: {e}")
        return {}


def get_new_unique_name(base_name: str, parent_dir: Path) -> str:
    """Generate a unique folder name by appending _new suffix if needed."""
    existing_names = {p.name for p in parent_dir.iterdir()}

    if base_name not in existing_names:
        return base_name

    attempt = 1
    while True:
        new_name = f"{base_name}_new{attempt}"
        if new_name not in existing_names:
            return new_name
        attempt += 1


def rename_version_folder(
    root_dir: Path, version_dir: Path, dry_run: bool = False
) -> tuple[str, bool]:
    """
    Rename a version_ folder.

    Returns: (status_message, is_renamed)
    """
    hparams_path = version_dir / "hparams.yaml"

    if not hparams_path.exists():
        return f"⚠ Skipped {version_dir.relative_to(root_dir)} (no hparams.yaml)", False

    hparams = load_hparams(hparams_path)
    conv_type = str(hparams.get("conv_type", "")).strip()

    if not conv_type:
        return f"⚠ Skipped {version_dir.relative_to(root_dir)} (no conv_type)", False

    new_base_name = get_new_name(conv_type, hparams)
    if new_base_name is None:
        return (
            f"⚠ Skipped {version_dir.relative_to(root_dir)} (invalid dataset or conv_type)",
            False,
        )

    parent_dir = version_dir.parent
    target_name = get_new_unique_name(new_base_name, parent_dir)
    target_path = parent_dir / target_name
    old_relative = version_dir.relative_to(root_dir)

    if dry_run:
        return f"DRY-RUN: Would rename {old_relative} → {target_name}", True

    try:
        shutil.move(str(version_dir), str(target_path))
        return f"✓ Renamed: {old_relative} → {target_name}", True
    except Exception as e:
        return f"⚠ Error renaming {old_relative}: {e}", False


def find_version_folders(root_dir: Path) -> list[Path]:
    """Find all version_ directories recursively."""
    if root_dir.is_file():
        print(f"Error: {root_dir} is a file, not a directory")
        return []

    version_folders = [
        item
        for item in root_dir.rglob("*")
        if item.is_dir() and item.name.startswith("version_")
    ]

    # Sort by version index for predictable order
    def extract_num(p: Path):
        parts = p.name.split("_")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    version_folders.sort(key=extract_num)
    return version_folders


def main():
    parser = argparse.ArgumentParser(
        description="Rename version_ folders in tensorboard logs based on hparams.yaml"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="logs",
        help="Root directory to scan for version_ folders (default: logs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without making changes",
    )

    args = parser.parse_args()
    root_dir = Path(args.root_dir).expanduser()

    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist")
        return 1

    if not root_dir.is_dir():
        print(f"Error: {root_dir} is not a directory")
        return 1

    version_folders = find_version_folders(root_dir)

    if not version_folders:
        print(f"No version_ directories found in {root_dir}")
        return 0

    print(f"Found {len(version_folders)} version_ directory(ies)")
    print("-" * 60)

    success_count = 0
    skip_count = 0

    for version_folder in version_folders:
        status, success = rename_version_folder(
            root_dir, version_folder, dry_run=args.dry_run
        )
        print(status)

        if success:
            success_count += 1
        else:
            skip_count += 1

    print("-" * 60)
    if args.dry_run:
        print(
            f"Dry-run complete. Would rename {success_count} folder(s), skip {skip_count}."
        )
    else:
        print(
            f"Done! Renamed {success_count} folder(s), skipped {skip_count} folder(s)."
        )

    return 0 if skip_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
