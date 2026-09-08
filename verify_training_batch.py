"""Load and describe one batch using the same preprocessing as the Hebrew runner."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from train_words_3000 import WordDataset, collate, collect_samples_from_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Verify a preprocessed Hebrew HTR batch")
    parser.add_argument("data_dirs", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--strip-whitespace", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = collect_samples_from_dirs(args.data_dirs, strip_whitespace=args.strip_whitespace)
    checked_dataset = WordDataset(
        samples,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
    )
    loader = DataLoader(
        checked_dataset,
        batch_size=min(args.batch_size, len(checked_dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    images, labels, datasets, paths = next(iter(loader))

    print("Batch shape:", tuple(images.shape))
    print("Pixel range:", float(images.min()), float(images.max()))
    print("Mirrored for RTL:", not args.no_mirror)
    for dataset_name, path, label in zip(datasets, paths, labels):
        print(f"- dataset={dataset_name} image={Path(path).name} label={label!r}")


if __name__ == "__main__":
    main()
