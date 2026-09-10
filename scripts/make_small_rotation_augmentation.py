#!/usr/bin/env python3
"""
Create a small-rotation augmentation copy of OCR/HTR image+txt pairs.

Default behavior matches the project layout used for Hebrew HTR training:
  input:  data/train/arielManual, data/train/RockManual, data/train/AgadaManual
  output: data/train_small_rotation_YYYYMMDD/<dataset-name>/

For every image that has a matching .txt label, the script writes one rotated
image and copies the label unchanged. File names are preserved.

Example:
  python scripts/make_small_rotation_augmentation.py \
    --repo /home/maxim/HTR-VT \
    --output data/train_small_rotation_20260816 \
    --degrees 2.0 \
    --seed 20260816
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_DATASETS = ["arielManual", "RockManual", "AgadaManual"]


def iter_pairs(dataset_dir: Path) -> Iterable[tuple[Path, Path]]:
    for image_path in sorted(dataset_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            yield image_path, txt_path


def rotate_image(src: Path, dst: Path, angle: float, fill: str = "white") -> None:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        rotated = im.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=fill,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        rotated.save(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create small-rotation OCR augmentation data.")
    parser.add_argument("--repo", default="/home/maxim/HTR-VT", help="Repository/data root containing data/.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, help="Dataset folder names under data/.")
    parser.add_argument("--output", default=None, help="Output folder, relative to repo or absolute.")
    parser.add_argument("--degrees", type=float, default=2.0, help="Maximum absolute rotation angle in degrees.")
    parser.add_argument("--seed", type=int, default=20260816, help="Random seed for deterministic angles.")
    parser.add_argument("--fill", default="white", help="Fill color for exposed corners.")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    data_root = repo / "data"
    out_root = Path(args.output) if args.output else data_root / f"train_small_rotation_{args.seed}"
    if not out_root.is_absolute():
        out_root = repo / out_root

    rng = random.Random(args.seed)
    total = 0
    for dataset_name in args.datasets:
        src_dir = data_root / dataset_name
        dst_dir = out_root / dataset_name
        if not src_dir.exists():
            print(f"skip missing dataset: {src_dir}")
            continue

        count = 0
        for image_path, txt_path in iter_pairs(src_dir):
            angle = rng.uniform(-args.degrees, args.degrees)
            rotate_image(image_path, dst_dir / image_path.name, angle, fill=args.fill)
            shutil.copy2(txt_path, dst_dir / txt_path.name)
            count += 1

        print(f"{dataset_name}: {count} pairs -> {dst_dir}")
        total += count

    print(f"done: {total} augmented pairs written under {out_root}")


if __name__ == "__main__":
    main()
