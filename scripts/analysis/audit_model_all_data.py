from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import HTR_VT
from train_words_3000 import (
    WordDataset,
    collect_samples,
    collate,
    evaluate,
    patch_forward_no_final_logit_norm,
)
from utils import utils


def severity(edit_distance, truth):
    ratio = edit_distance / max(1, len(truth))
    if edit_distance >= 3 or ratio >= 0.6:
        return "hard"
    if edit_distance >= 2 or ratio >= 0.3:
        return "medium"
    if edit_distance:
        return "minor"
    return "exact"


def main():
    parser = argparse.ArgumentParser(description="Audit a checkpoint on every labeled word")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
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

    samples = collect_samples(args.data_dir)
    dataset = WordDataset(samples, width, height, mirror, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = HTR_VT.create_model(nb_cls=len(alphabet) + 1, img_size=[height, width]).to(device)
    patch_forward_no_final_logit_norm(model)
    model.load_state_dict(checkpoint["model"])
    converter = utils.CTCLabelConverter(alphabet)
    criterion = torch.nn.CTCLoss(reduction="mean", zero_infinity=True)
    result = evaluate(model, loader, converter, criterion, device, device.type == "cuda")

    rows = []
    for row in result["predictions"]:
        enriched = dict(row)
        enriched["truth_length"] = len(row["truth"])
        enriched["error_ratio"] = row["edit_distance"] / max(1, len(row["truth"]))
        enriched["severity"] = severity(row["edit_distance"], row["truth"])
        rows.append(enriched)
    rows.sort(key=lambda row: (row["edit_distance"], row["error_ratio"]), reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ("severity", "dataset", "image", "truth", "prediction", "edit_distance", "truth_length", "error_ratio")
    with (args.output_dir / "all_predictions_ranked.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    hard_rows = [row for row in rows if row["severity"] == "hard"]
    with (args.output_dir / "hard_errors.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hard_rows)

    grouped = defaultdict(lambda: {"words": 0, "exact": 0, "characters": 0, "edit_distance": 0, "hard": 0})
    for row in rows:
        stats = grouped[row["dataset"]]
        stats["words"] += 1
        stats["exact"] += int(row["edit_distance"] == 0)
        stats["characters"] += row["truth_length"]
        stats["edit_distance"] += row["edit_distance"]
        stats["hard"] += int(row["severity"] == "hard")
    for stats in grouped.values():
        stats["word_accuracy"] = stats["exact"] / stats["words"]
        stats["cer"] = stats["edit_distance"] / max(1, stats["characters"])

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "device": str(device),
        "words": len(rows),
        "exact": sum(row["edit_distance"] == 0 for row in rows),
        "word_accuracy": result["word_accuracy"],
        "cer": result["cer"],
        "hard_errors": len(hard_rows),
        "medium_errors": sum(row["severity"] == "medium" for row in rows),
        "minor_errors": sum(row["severity"] == "minor" for row in rows),
        "by_dataset": dict(grouped),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
