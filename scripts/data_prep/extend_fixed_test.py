from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import random
from collections import Counter
from pathlib import Path

from train_words_3000 import DATASETS, collect_samples, read_fixed_test_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(description="Extend the persistent fixed test set evenly")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/fixed_hebrew_test_words.csv"))
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=124)
    args = parser.parse_args()

    samples = collect_samples(args.data_dir)
    test = read_fixed_test_manifest(args.manifest, samples)
    if len(test) > args.size:
        raise RuntimeError(f"Manifest already has {len(test)} rows; refusing to shrink it")

    rng = random.Random(args.seed)
    selected = {item["image"] for item in test}
    counts = Counter(item["dataset"] for item in test)
    added = []
    while len(test) < args.size:
        dataset = min(DATASETS, key=lambda name: (counts[name], DATASETS.index(name)))
        candidates = [
            item for item in samples
            if item["dataset"] == dataset and item["image"] not in selected
        ]
        if not candidates:
            raise RuntimeError(f"No unused candidates remain in {dataset}")
        item = rng.choice(candidates)
        test.append(item)
        added.append(item)
        selected.add(item["image"])
        counts[dataset] += 1

    write_manifest(args.manifest, test)
    print(f"test={len(test)} counts={dict(counts)}")
    for item in added:
        print(f"added {item['dataset']}: {item['image']} ({item['label']})")


if __name__ == "__main__":
    main()
