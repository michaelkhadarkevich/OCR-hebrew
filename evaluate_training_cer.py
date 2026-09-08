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
    collate,
    collect_samples_from_dirs,
    evaluate,
    patch_forward_no_final_logit_norm,
)
from utils import utils


def main():
    parser = argparse.ArgumentParser(
        description="Calculate CER of a checkpoint on the training folders recorded in its config"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    train_dirs = config.get("train_dirs")
    if not train_dirs:
        raise RuntimeError("Checkpoint config does not contain explicit train_dirs")

    strip_whitespace = bool(config.get("strip_whitespace", False))
    samples = collect_samples_from_dirs(
        [Path(path) for path in train_dirs], strip_whitespace=strip_whitespace
    )
    alphabet = checkpoint["alphabet"]
    width = int(config.get("width", 512))
    height = int(config.get("height", 64))
    mirror = bool(config.get("mirror", True))

    dataset = WordDataset(samples, width, height, mirror, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    model = HTR_VT.create_model(nb_cls=len(alphabet) + 1, img_size=[height, width]).to(device)
    patch_forward_no_final_logit_norm(model)
    model.load_state_dict(checkpoint["model"])
    converter = utils.CTCLabelConverter(alphabet)
    criterion = torch.nn.CTCLoss(reduction="mean", zero_infinity=True)
    result = evaluate(model, loader, converter, criterion, device, amp_enabled=False)

    by_dataset = defaultdict(lambda: {"samples": 0, "exact": 0, "characters": 0, "edits": 0})
    for row in result["predictions"]:
        stats = by_dataset[row["dataset"]]
        stats["samples"] += 1
        stats["exact"] += int(row["truth"] == row["prediction"])
        stats["characters"] += len(row["truth"])
        stats["edits"] += int(row["edit_distance"])

    for stats in by_dataset.values():
        stats["cer"] = stats["edits"] / stats["characters"]
        stats["exact_accuracy"] = stats["exact"] / stats["samples"]

    total_characters = sum(len(row["truth"]) for row in result["predictions"])
    total_edits = sum(int(row["edit_distance"]) for row in result["predictions"])
    exact_samples = sum(
        row["truth"] == row["prediction"] for row in result["predictions"]
    )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "device": str(device),
        "strip_whitespace": strip_whitespace,
        "augmentations": False,
        "samples": len(samples),
        "characters": total_characters,
        "edit_distance": total_edits,
        "train_cer": total_edits / total_characters,
        "character_accuracy": 1.0 - total_edits / total_characters,
        "exact_samples": exact_samples,
        "exact_accuracy": exact_samples / len(samples),
        "ctc_loss": result["loss"],
        "by_dataset": dict(by_dataset),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "image", "truth", "prediction", "edit_distance"),
        )
        writer.writeheader()
        writer.writerows(result["predictions"])
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
