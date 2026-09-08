"""Generate the small, verified English result figures published in the repository."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path("docs/results")
FINAL_RESULTS = RESULTS_DIR / "verified_baseline_results.csv"
CURVES = RESULTS_DIR / "verified_validation_curves.csv"
LOCAL_RUNS = {
    "Words only": [Path("output/only_words_four_datasets_20000_rotation/run/metrics.csv")],
    "Lines only": [Path("output/manual_lines_10000_rotation/run/metrics.csv")],
    "Lines + words": [
        Path("output/manual_lines_plus_words_10000_rotation/run/metrics.csv"),
        Path("output/manual_lines_plus_words_continued_20000/run/metrics.csv"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate verified public HTR result figures")
    parser.add_argument(
        "--refresh-curves-from-output",
        action="store_true",
        help="Recreate the compact validation CSV from ignored local run outputs",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def refresh_curves():
    rows = []
    for label, paths in LOCAL_RUNS.items():
        source_rows = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing local metric file: {path}")
            source_rows.extend(read_csv(path))
        source_rows.sort(key=lambda row: int(row["step"]))
        for row in source_rows:
            step = int(row["step"])
            if step == 1 or step % 100 == 0:
                rows.append(
                    {
                        "training_data": label,
                        "step": step,
                        "validation_cer_percent": 100.0 * float(row["test_cer"]),
                        "validation_exact_line_accuracy_percent": 100.0
                        * float(row["test_word_accuracy"]),
                    }
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CURVES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_final_results():
    rows = read_csv(FINAL_RESULTS)
    labels = [row["training_data"] for row in rows]
    cer = [float(row["final_test_cer_percent"]) for row in rows]
    exact = [float(row["exact_line_accuracy_percent"]) for row in rows]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar(labels, cer, color=colors)
    axes[0].set_title("Final-test character error rate")
    axes[0].set_ylabel("CER (%) — lower is better")
    axes[0].set_ylim(0, max(cer) * 1.25)
    for index, value in enumerate(cer):
        axes[0].text(index, value + 0.25, f"{value:.2f}%", ha="center")

    axes[1].bar(labels, exact, color=colors)
    axes[1].set_title("Final-test exact-line accuracy")
    axes[1].set_ylabel("Exact lines (%) — higher is better")
    axes[1].set_ylim(0, 60)
    for index, (value, row) in enumerate(zip(exact, rows)):
        axes[1].text(
            index,
            value + 1.0,
            f"{row['exact_lines']}/{row['total_lines']}\n({value:.2f}%)",
            ha="center",
        )

    figure.suptitle("Verified results on 32 held-out lines (whitespace excluded)")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "verified_final_test_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_validation_curves():
    rows = read_csv(CURVES)
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    for label in ("Words only", "Lines only", "Lines + words"):
        selected = [
            row
            for row in rows
            if row["training_data"] == label and int(row["step"]) >= 1000
        ]
        axis.plot(
            [int(row["step"]) for row in selected],
            [float(row["validation_cer_percent"]) for row in selected],
            label=label,
            linewidth=1.8,
        )
    axis.set_title("Validation CER after the first 1,000 optimizer steps")
    axis.set_xlabel("Optimizer steps")
    axis.set_ylabel("Validation CER (%) — lower is better")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "verified_validation_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.refresh_curves_from_output:
        refresh_curves()
    if not CURVES.is_file():
        raise FileNotFoundError(
            f"Missing {CURVES}. Run once with --refresh-curves-from-output on the training machine."
        )
    plot_final_results()
    plot_validation_curves()
    print(f"Generated verified result figures in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
