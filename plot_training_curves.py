import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No metric rows in {path}")
    return {
        "step": [int(row["step"]) for row in rows],
        "train_loss": [float(row["train_loss"]) for row in rows],
        "validation_loss": [float(row["test_loss"]) for row in rows],
        "validation_cer": [100.0 * float(row["test_cer"]) for row in rows],
        "validation_accuracy": [100.0 * float(row["test_word_accuracy"]) for row in rows],
    }


def plot_single(label, metrics, destination):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(metrics["step"], metrics["train_loss"], label="Training loss", linewidth=2)
    axes[0].plot(
        metrics["step"], metrics["validation_loss"], label="Validation loss", linewidth=2
    )
    axes[0].set_title(f"{label}: training and validation loss")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("CTC loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        metrics["step"], metrics["validation_cer"], label="Validation CER", linewidth=2
    )
    axes[1].plot(
        metrics["step"],
        metrics["validation_accuracy"],
        label="Exact-sample accuracy",
        linewidth=2,
    )
    axes[1].set_title(f"{label}: validation metrics")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Percent")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Plot step-based HTR training curves")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="Label and run directory containing metrics.csv; repeat for multiple runs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for specification in args.run:
        if "=" not in specification:
            raise ValueError(f"Expected LABEL=RUN_DIR, got {specification!r}")
        label, raw_path = specification.split("=", 1)
        safe_label = "_".join(label.lower().split())
        run_dir = Path(raw_path)
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
        if (config.get("ema_decay", 0) > 0 or config.get("warm_up_steps", 0) > 0) and (
            config.get("training_recipe_version") != "global-cosine-bound-ema-v1"
        ):
            print(f"WARNING: {run_dir} predates EMA/schedule verification; historical metrics only.")
            label += " [legacy protocol]"
        metrics = read_metrics(run_dir / "metrics.csv")
        runs.append((label, metrics))
        plot_single(label, metrics, args.output_dir / f"{safe_label}_curves.png")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, metrics in runs:
        axes[0].plot(metrics["step"], metrics["validation_cer"], label=label, linewidth=2)
        axes[1].plot(
            metrics["step"], metrics["validation_accuracy"], label=label, linewidth=2
        )
    axes[0].set_title("Validation CER comparison")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("CER (%)")
    axes[1].set_title("Validation exact-sample accuracy comparison")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "three_experiments_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
