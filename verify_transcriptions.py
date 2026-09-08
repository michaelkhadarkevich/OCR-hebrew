"""Validate recursively stored image/transcription pairs before HTR training."""

import argparse
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check image/TXT pairs and UTF-8 transcriptions in one or more datasets"
    )
    parser.add_argument("data_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--require-no-whitespace",
        action="store_true",
        help="Report labels containing whitespace (useful for word-only datasets)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    failures = []
    labels = []
    image_count = 0

    for root in args.data_dirs:
        if not root.is_dir():
            failures.append(f"Dataset directory does not exist: {root}")
            continue

        images = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        image_count += len(images)
        image_stems = {path.with_suffix("") for path in images}

        for image_path in images:
            label_path = image_path.with_suffix(".txt")
            if not label_path.is_file():
                failures.append(f"Missing label: {image_path}")
                continue
            try:
                label = label_path.read_text(encoding="utf-8-sig").strip()
            except UnicodeDecodeError as error:
                failures.append(f"Invalid UTF-8: {label_path} ({error})")
                continue
            if not label:
                failures.append(f"Empty label: {label_path}")
                continue
            if args.require_no_whitespace and any(character.isspace() for character in label):
                failures.append(f"Whitespace in word label: {label_path}")
            labels.append(label)

        for label_path in sorted(root.rglob("*.txt")):
            if label_path.with_suffix("") not in image_stems:
                failures.append(f"Missing image for label: {label_path}")

    alphabet = Counter(character for label in labels for character in label)
    print(f"Datasets: {len(args.data_dirs)}")
    print(f"Images: {image_count}")
    print(f"Valid non-empty labels: {len(labels)}")
    print(f"Unique characters: {len(alphabet)}")
    if alphabet:
        print("Alphabet:", "".join(sorted(alphabet)))

    if failures:
        print(f"Problems: {len(failures)}")
        for failure in failures[:50]:
            print("-", failure)
        if len(failures) > 50:
            print(f"- ... and {len(failures) - 50} more")
        raise SystemExit(1)

    print("Validation passed: every image has a non-empty UTF-8 label and every label has an image.")


if __name__ == "__main__":
    main()
