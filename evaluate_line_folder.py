import argparse
import csv
import json
from pathlib import Path

import editdistance
import torch
from torch.utils.data import DataLoader

from model import HTR_VT
from train_words_3000 import WordDataset, collate, patch_forward_no_final_logit_norm
from utils import utils


def compact(text):
    return "".join(text.split())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Evaluate an HTR checkpoint on recursively stored labeled lines")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    samples = []
    for image in sorted(args.data_dir.rglob("*.png")):
        label_file = image.with_suffix(".txt")
        if not label_file.exists():
            continue
        samples.append(
            {
                "dataset": args.data_dir.name,
                "image": str(image.resolve()),
                "label_file": str(label_file.resolve()),
                "label": label_file.read_text(encoding="utf-8-sig").strip(),
            }
        )
    if not samples:
        raise RuntimeError("No PNG/TXT pairs found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    width = int(config.get("width", 512))
    height = int(config.get("height", 64))
    mirror = bool(config.get("mirror", True))
    alphabet = checkpoint["alphabet"]

    dataset = WordDataset(samples, width, height, mirror, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = HTR_VT.create_model(nb_cls=len(alphabet) + 1, img_size=[height, width]).to(device)
    patch_forward_no_final_logit_norm(model)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    converter = utils.CTCLabelConverter(alphabet)

    rows = []
    for images, labels, datasets, paths in loader:
        logits = model(images.to(device))
        sizes = torch.full((images.size(0),), logits.size(1), dtype=torch.int32)
        indices = logits.float().argmax(2).reshape(-1).cpu()
        predictions = converter.decode(indices, sizes)
        for path, truth, prediction in zip(paths, labels, predictions):
            truth_compact = compact(truth)
            prediction_compact = compact(prediction)
            rows.append(
                {
                    "image": path,
                    "truth": truth,
                    "prediction": prediction,
                    "raw_edit_distance": editdistance.eval(prediction, truth),
                    "truth_without_spaces": truth_compact,
                    "prediction_without_spaces": prediction_compact,
                    "compact_edit_distance": editdistance.eval(prediction_compact, truth_compact),
                    "raw_exact": prediction == truth,
                    "compact_exact": prediction_compact == truth_compact,
                }
            )

    raw_chars = sum(len(row["truth"]) for row in rows)
    compact_chars = sum(len(row["truth_without_spaces"]) for row in rows)
    raw_ed = sum(row["raw_edit_distance"] for row in rows)
    compact_ed = sum(row["compact_edit_distance"] for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "device": str(device),
        "lines": len(rows),
        "raw_exact_lines": sum(row["raw_exact"] for row in rows),
        "raw_line_accuracy": sum(row["raw_exact"] for row in rows) / len(rows),
        "raw_cer": raw_ed / raw_chars,
        "compact_exact_lines": sum(row["compact_exact"] for row in rows),
        "compact_line_accuracy": sum(row["compact_exact"] for row in rows) / len(rows),
        "compact_cer": compact_ed / compact_chars,
        "model_has_space_class": " " in alphabet,
        "model_width": width,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
