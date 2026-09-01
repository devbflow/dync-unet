"""
Script to rename version_ folders in tensorboard logs based on hparams.yaml configuration.

Naming schemes:
- att-dynamic -> att_<initial_planes>_<num_layers>_<K>_<lr>_<dataset>_<data_type>
- stat -> stat_<initial_planes>_<num_layers>_<lr>_<dataset>_<data_type>
- wp-dynamic -> wp_<initial_planes>_<num_layers>_<kernel_gen_hidden>_<lr>_<dataset>_<data_type>
- wm-dynamic -> wm_<initial_planes>_<num_layers>_<kernel_gen_hidden>_<lr>_<dataset>_<data_type>
"""

import argparse
import shutil
from pathlib import Path

# Dataset mapping
DATASET_MAP = {
    "OpenEarthMap-SAR": "oem",
    "whu-opt-sar": "whu",
}


def extract_dataset_from_data_dir(data_dir: str) -> str | None:
    """Extract dataset name from data_dir path."""
    if not data_dir or data_dir.strip() == "":
        return None

    # Normalize the path for comparison
    normalized = data_dir.strip().rstrip("/")

    for full_dataset_name, short_name in DATASET_MAP.items():
        if full_dataset_name in normalized:
            return short_name

    return None


def parse_lr(lr_str) -> str:
    """Parse learning rate and format as string."""
    # Convert float to scientific notation with 1e-3 or 1e-2
    try:
        lr_float = float(lr_str)
        if abs(lr_float) < 0.001:
            return f"{lr_float:.1e}".replace("e+", "e+").replace("e-", "e-")
        else:
            # Format as scientific notation with 1e-x format
            import math

            exponent = math.floor(math.log10(abs(lr_float)))
            if lr_float < 0:
                return f"{lr_float:.2e}"
            elif abs(lr_float) >= 10**exponent:
                return f"{lr_float:.1e}"
            else:
                return f"1{abs(lr_float)}{exponent:+d}"
    except (ValueError, TypeError):
        # If parsing fails, return as string
        if lr_str is None:
            return "null"
        return str(lr_str)


def get_new_name(conv_type: str, hparams: dict) -> str | None:
    """Generate new folder name based on conv_type and hparams."""

    # Extract base parameters
    initial_planes = int(hparams.get("initial_planes", 0))
    num_layers = int(hparams.get("num_layers", 0))
    lr_str = parse_lr(hparams.get("lr"))
    dataset = extract_dataset_from_data_dir(hparams.get("data_dir", ""))
    data_type = str(hparams.get("data_type", "sar")).lower()

    # Skip if no valid dataset found
    if not dataset:
        return None

    # Determine naming pattern based on conv_type
    if conv_type == "att-dynamic":
        initial_planes_str = f"{initial_planes}"
        num_layers_str = f"{num_layers}"

        # K is only included when not null
        k_val = hparams.get("K")
        if k_val and k_val is not None:
            k_str = str(k_val)
        else:
            k_str = ""  # Omitted

        lr_str = parse_lr(hparams.get("lr"))

        name_parts = [initial_planes_str, num_layers_str]
        if k_str:
            name_parts.append(k_str)
        name_parts.append(lr_str)
        name_parts.append(dataset)
        name_parts.append(data_type)

        return "att_" + "_".join(name_parts)

    elif conv_type == "static":
        initial_planes_str = f"{initial_planes}"
        num_layers_str = f"{num_layers}"
        lr_str = parse_lr(hparams.get("lr"))

        name_parts = [initial_planes_str, num_layers_str, lr_str, dataset, data_type]
        return "stat_" + "_".join(name_parts)

    elif conv_type == "wp-dynamic":
        initial_planes_str = f"{initial_planes}"
        num_layers_str = f"{num_layers}"

        # kernel_gen_hidden is only included when not null
        kh_val = hparams.get("kernel_gen_hidden")
        if kh_val and kh_val is not None:
            kh_str = str(kh_val)
        else:
            kh_str = ""  # Omitted

        lr_str = parse_lr(hparams.get("lr"))

        name_parts = [initial_planes_str, num_layers_str]
        if kh_str:
            name_parts.append(kh_str)
        name_parts.append(lr_str)
        name_parts.append(dataset)
        name_parts.append(data_type)

        return "wp_" + "_".join(name_parts)

    elif conv_type == "wm-dynamic":
        initial_planes_str = f"{initial_planes}"
        num_layers_str = f"{num_layers}"

        # kernel_gen_hidden is only included when not null
        kh_val = hparams.get("kernel_gen_hidden")
        if kh_val and kh_val is not None:
            kh_str = str(kh_val)
        else:
            kh_str = ""  # Omitted

        lr_str = parse_lr(hparams.get("lr"))

        name_parts = [initial_planes_str, num_layers_str]
        if kh_str:
            name_parts.append(kh_str)
        name_parts.append(lr_str)
        name_parts.append(dataset)
        name_parts.append(data_type)

        return "wm_" + "_".join(name_parts)

    else:
        print(f"  Unknown conv_type: {conv_type}")
        return None


def load_hparams(hparams_path: Path) -> dict:
    """Load hparams.yaml file."""
    try:
        import yaml

        with open(hparams_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        print("  Warning: PyYAML not installed. Using basic YAML parsing.")
        # Basic YAML parsing fallback
        hparams = {}
        try:
            with open(hparams_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line and not line.startswith("#"):
                        key, _, value = line.partition(":")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Try to convert common types
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


def get_new_unique_name(base_name: str, parent_dir: Path) -> tuple[str, bool]:
    """Generate a unique name by adding _new suffix if needed."""
    existing = list(parent_dir.iterdir())

    if base_name not in [p.name for p in existing]:
        return base_name, False

    # Name exists, add _new suffix
    attempt = 1
    while True:
        new_name = f"{base_name}_new{attempt}"
        if new_name not in [p.name for p in existing]:
            return new_name, True
        attempt += 1


def rename_version_folder(root_dir: Path, version_dir: Path) -> tuple[str | None, bool]:
    """
    Rename a version_ folder.

    Returns: (success_status_message, was_skipped)
    - success_status_message: "✓ Renamed: <old> → <new>" or similar
    - was_skipped: True if the folder was skipped for any reason
    """

    hparams_path = version_dir / "hparams.yaml"

    # Check if hparams.yaml exists
    if not hparams_path.exists():
        return f"⚠ Skipped {version_dir.relative_to(root_dir)} (no hparams.yaml)", True

    # Load hparams
    hparams = load_hparams(hparams_path)

    # Extract conv_type
    conv_type = str(hparams.get("conv_type", "")).replace(
        "-", "dynamic"
    )  # e.g., "att-dynamic" -> "att-dynamic"
    conv_type_original = str(hparams.get("conv_type", ""))

    if not conv_type:
        return f"⚠ Skipped {version_dir.relative_to(root_dir)} (no conv_type)", True

    # Generate new name
    new_name = get_new_name(conv_type_original, hparams)

    if new_name is None:
        return (
            f"⚠ Skipped {version_dir.relative_to(root_dir)} (no valid dataset in data_dir)",
            True,
        )

    # Check for conflicts and resolve
    parent_dir = version_dir.parent
    target_path = parent_dir / new_name

    existing_items = list(parent_dir.iterdir())
    if new_name not in [p.name for p in existing_items]:
        # No conflict, direct rename
        old_relative = version_dir.relative_to(root_dir)
        print(f"✓ Renamed: {old_relative} → {new_name}")
        shutil.move(str(version_dir), str(target_path))
        return f"✓ Renamed: {old_relative} → {new_name}", False
    else:
        # Conflict detected, find unique name
        for attempt in range(1, 6):
            new_name_unique = f"{new_name}_new{attempt}"
            if new_name_unique not in [p.name for p in existing_items]:
                target_path = parent_dir / new_name_unique
                old_relative = version_dir.relative_to(root_dir)
                print(f"✓ Renamed: {old_relative} → {new_name_unique}")
                shutil.move(str(version_dir), str(target_path))
                return f"✓ Renamed: {old_relative} → {new_name_unique}", False

        # Still conflict after 5 attempts
        last_name = f"{new_name}_new5"
        old_relative = version_dir.relative_to(root_dir)
        print(
            f"⚠ Skipped {old_relative} (conflict with existing folders, reached max retries)"
        )
        return f"⚠ Skipped {old_relative} (conflict resolution failed)", True


def find_version_folders(root_dir: Path) -> list:
    """Find all version_ directories recursively."""
    version_folders = []

    if root_dir.is_file():
        print(f"Error: {root_dir} is a file, not a directory")
        return []

    for item in root_dir.rglob("*"):
        if item.is_dir() and item.name.startswith("version_"):
            version_folders.append(item)

    # Sort by name for consistent processing order
    version_folders.sort(
        key=lambda p: int(p.name.split("_")[1]) if p.name.count("_") > 0 else 0
    )

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

    # Find all version_ folders
    version_folders = find_version_folders(root_dir)

    if not version_folders:
        print(f"No version_ directories found in {root_dir}")
        return 0

    print(f"Found {len(version_folders)} version_ directory(ies)")
    print("-" * 60)

    # Process each version folder
    success_count = 0
    skip_count = 0

    for version_folder in version_folders:
        old_relative = version_folder.relative_to(root_dir)

        if args.dry_run:
            status, _ = rename_version_folder(root_dir, version_folder)
            # For dry-run, show what would happen without actually renaming
            hparams_path = version_folder / "hparams.yaml"
            if hparams_path.exists():
                print(f"DRY-RUN: {status}")
                hparams = load_hparams(hparams_path)
                conv_type = str(hparams.get("conv_type", "")).replace("-", "-dynamic")
                new_name = get_new_name(conv_type.replace("-dynamic", ""), hparams)
                if new_name:
                    parent_dir = version_folder.parent
                    target_path = parent_dir / new_name
                    if new_name not in [p.name for p in parent_dir.iterdir()]:
                        print(f"DRY-RUN: Would rename {old_relative} → {new_name}")
            skip_count += 1
        else:
            status, skipped = rename_version_folder(root_dir, version_folder)
            if "Renamed" in status:
                success_count += 1
            elif "Skipped" in status and not status.startswith("⚠ Skipped"):
                # Skip for conflict (not invalid hparams)
                skip_count += 1

        print()  # Empty line for readability

    print("-" * 60)
    if args.dry_run:
        print(f"Dry-run complete. Would rename {success_count} folder(s)")
        print(f"Skipped {skip_count} folder(s) due to invalid configuration")
    else:
        print(
            f"Done! Renamed {success_count} folder(s), skipped {skip_count} folder(s)"
        )

    return 0 if skip_count == 0 else 1


if __name__ == "__main__":
    exit(main())
