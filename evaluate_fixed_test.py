import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import HTR_VT
from train_words_3000 import WordDataset, collate, evaluate, patch_forward_no_final_logit_norm
from utils import utils


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved HTR-VT checkpoint on a fixed manifest")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("fixed_hebrew_test_words.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    alphabet = checkpoint["alphabet"]
    config = checkpoint.get("config", {})
    width = int(config.get("width", 512))
    height = int(config.get("height", 64))
    mirror = bool(config.get("mirror", True))

    with args.manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    samples = []
    for row in manifest_rows:
        image = Path(row["image"])
        label_file = Path(row["label_file"])
        if not image.exists() or not label_file.exists():
            raise FileNotFoundError(f"Missing test pair: {image} / {label_file}")
        samples.append(
            {
                "dataset": row["dataset"],
                "image": str(image.resolve()),
                "label_file": str(label_file.resolve()),
                "label": label_file.read_text(encoding="utf-8-sig").strip(),
            }
        )

    dataset = WordDataset(samples, width, height, mirror, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = HTR_VT.create_model(nb_cls=len(alphabet) + 1, img_size=[height, width]).to(device)
    patch_forward_no_final_logit_norm(model)
    model.load_state_dict(checkpoint["model"])
    converter = utils.CTCLabelConverter(alphabet)
    criterion = torch.nn.CTCLoss(reduction="mean", zero_infinity=True)
    result = evaluate(model, loader, converter, criterion, device, device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "image", "truth", "prediction", "edit_distance"),
        )
        writer.writeheader()
        writer.writerows(result["predictions"])

    by_dataset = {}
    for row in result["predictions"]:
        stats = by_dataset.setdefault(
            row["dataset"], {"words": 0, "exact": 0, "characters": 0, "edit_distance": 0}
        )
        stats["words"] += 1
        stats["exact"] += int(row["truth"] == row["prediction"])
        stats["characters"] += len(row["truth"])
        stats["edit_distance"] += row["edit_distance"]
    for stats in by_dataset.values():
        stats["word_accuracy"] = stats["exact"] / stats["words"]
        stats["cer"] = stats["edit_distance"] / stats["characters"]

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "manifest": str(args.manifest.resolve()),
        "test_words": len(samples),
        "loss": result["loss"],
        "cer": result["cer"],
        "word_accuracy": result["word_accuracy"],
        "by_dataset": by_dataset,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
