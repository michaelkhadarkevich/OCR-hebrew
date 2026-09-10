from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Create a side-by-side preview of source images and mirrored model inputs."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create an RTL preprocessing preview")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/rtl_preview.png"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--width", type=int, default=480)
    return parser.parse_args()


def fit_width(image, width):
    image = image.convert("L")
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def main():
    args = parse_args()
    images = [
        path
        for path in sorted(args.data_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ][: args.count]
    if not images:
        raise SystemExit(f"No images found under {args.data_dir}")

    rows = []
    for path in images:
        with Image.open(path) as source:
            original = fit_width(source, args.width)
        mirrored = ImageOps.mirror(original)
        row_height = max(original.height, mirrored.height) + 34
        row = Image.new("L", (args.width * 2 + 20, row_height), 255)
        row.paste(original, (0, 28))
        row.paste(mirrored, (args.width + 20, 28))
        draw = ImageDraw.Draw(row)
        draw.text((4, 6), f"Source: {path.name}", fill=0)
        draw.text((args.width + 24, 6), "Model input: horizontal mirror", fill=0)
        rows.append(row)

    preview = Image.new(
        "L", (max(row.width for row in rows), sum(row.height for row in rows)), 255
    )
    y = 0
    for row in rows:
        preview.paste(row, (0, y))
        y += row.height

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.output)
    print(f"Saved {len(rows)} source/mirrored pairs to {args.output}")


if __name__ == "__main__":
    main()
