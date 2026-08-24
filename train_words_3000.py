import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import editdistance
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from model import HTR_VT
from utils import utils


DATASETS = (
    "arielOnlyWord",
    "agadaOnlyWord",
    "RockOnlyWord",
    "phisicsTigulOnlyWords",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train HTR-VT on the three Hebrew word datasets")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/hebrew_words_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=int, default=90)
    parser.add_argument(
        "--fixed-test-manifest",
        type=Path,
        default=Path("fixed_hebrew_test_words.csv"),
        help="Persistent test split reused across runs; created once when missing",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rotation-degrees", type=float, default=0.0)
    parser.add_argument("--rotation-probability", type=float, default=0.5)
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror RTL word images for CTC")
    parser.add_argument("--smoke", action="store_true", help="Run one train/eval step without replacing final outputs")
    return parser.parse_args()


def collect_samples(data_dir):
    samples = []
    for dataset_name in DATASETS:
        root = data_dir / dataset_name
        for image_path in sorted(root.rglob("*.png")):
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                continue
            label = label_path.read_text(encoding="utf-8-sig").strip()
            if not label or " " in label:
                continue
            samples.append(
                {
                    "dataset": dataset_name,
                    "image": str(image_path.resolve()),
                    "label_file": str(label_path.resolve()),
                    "label": label,
                }
            )
    if not samples:
        raise RuntimeError("No paired PNG/TXT word samples were found")
    return samples


def read_fixed_test_manifest(path, samples):
    by_path = {str(Path(item["image"]).resolve()): item for item in samples}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    test = []
    missing = []
    for row in rows:
        key = str(Path(row["image"]).resolve())
        if key not in by_path:
            missing.append(key)
        else:
            test.append(by_path[key])
    if missing:
        raise RuntimeError(f"Fixed test manifest contains {len(missing)} missing samples; first: {missing[0]}")
    if len({item["image"] for item in test}) != len(test):
        raise RuntimeError("Fixed test manifest contains duplicate samples")
    return test


def split_samples(samples, test_size, seed, fixed_manifest):
    if fixed_manifest.exists():
        test = read_fixed_test_manifest(fixed_manifest, samples)
        if len(test) < 30:
            raise RuntimeError("The fixed test set must contain at least 30 words")
    else:
        if test_size < 30:
            raise ValueError("The fixed test set must contain at least 30 words")
        base, remainder = divmod(test_size, len(DATASETS))
        requested = {name: base + int(index < remainder) for index, name in enumerate(DATASETS)}
        rng = random.Random(seed)
        test = []
        for dataset_name in DATASETS:
            candidates = [item for item in samples if item["dataset"] == dataset_name]
            count = requested[dataset_name]
            if len(candidates) < count:
                raise RuntimeError(f"{dataset_name} has fewer than {count} usable samples")
            test.extend(rng.sample(candidates, count))
        write_manifest(fixed_manifest, test)
    test_paths = {item["image"] for item in test}
    train = [item for item in samples if item["image"] not in test_paths]
    return train, test


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "image", "label_file", "label"))
        writer.writeheader()
        for row in rows:
            portable = dict(row)
            for field in ("image", "label_file"):
                portable[field] = Path(os.path.relpath(row[field], Path.cwd())).as_posix()
            writer.writerow(portable)


class WordDataset(Dataset):
    def __init__(
        self,
        samples,
        width,
        height,
        mirror,
        augment=False,
        rotation_degrees=0.0,
        rotation_probability=0.5,
    ):
        self.samples = samples
        self.width = width
        self.height = height
        self.mirror = mirror
        self.augment = augment
        self.rotation_degrees = rotation_degrees
        self.rotation_probability = rotation_probability

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        with Image.open(item["image"]) as source:
            image = source.convert("L")
        if (
            self.augment
            and self.rotation_degrees > 0
            and random.random() < self.rotation_probability
        ):
            image = image.rotate(
                random.uniform(-self.rotation_degrees, self.rotation_degrees),
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor="white",
            )
        if self.mirror:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.augment:
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.25:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        new_width = min(self.width, max(1, round(image.width * self.height / image.height)))
        image = image.resize((new_width, self.height), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (self.width, self.height), 255)
        canvas.paste(image, (0, 0))
        pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).clone()
        tensor = pixels.reshape(self.height, self.width).unsqueeze(0).float().div_(255.0)
        return tensor, item["label"], item["dataset"], item["image"]


def collate(batch):
    images, labels, datasets, paths = zip(*batch)
    return torch.stack(images), list(labels), list(datasets), list(paths)


def patch_forward_no_final_logit_norm(model):
    def forward(x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        x = model.layer_norm(x)
        x = model.patch_embed(x)
        batch, channels, _, _ = x.shape
        x = x.view(batch, channels, -1).permute(0, 2, 1)
        if use_masking:
            x = model.random_masking(x, mask_ratio, max_span_length)
        x = x + model.pos_embed
        for block in model.blocks:
            x = block(x)
        return model.head(model.norm(x))

    model.forward = forward


def ctc_loss(model, images, labels, converter, criterion, device, amp_enabled):
    images = images.to(device, non_blocking=True)
    targets, target_lengths = converter.encode(labels)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        logits = model(images)
        time_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.int32, device=device)
        log_probs = logits.float().permute(1, 0, 2).log_softmax(2)
        loss = criterion(log_probs, targets, time_lengths, target_lengths)
    return loss, logits


@torch.inference_mode()
def evaluate(model, loader, converter, criterion, device, amp_enabled):
    model.eval()
    total_loss = 0.0
    batches = 0
    total_ed = 0
    total_chars = 0
    correct_words = 0
    rows = []
    for images, labels, datasets, paths in loader:
        loss, logits = ctc_loss(model, images, labels, converter, criterion, device, amp_enabled)
        sizes = torch.full((images.size(0),), logits.size(1), dtype=torch.int32)
        indices = logits.float().argmax(2).reshape(-1).cpu()
        predictions = converter.decode(indices, sizes)
        total_loss += float(loss)
        batches += 1
        for dataset_name, path, truth, prediction in zip(datasets, paths, labels, predictions):
            distance = editdistance.eval(prediction, truth)
            total_ed += distance
            total_chars += len(truth)
            correct_words += int(prediction == truth)
            rows.append(
                {
                    "dataset": dataset_name,
                    "image": path,
                    "truth": truth,
                    "prediction": prediction,
                    "edit_distance": distance,
                }
            )
    model.train()
    return {
        "loss": total_loss / max(1, batches),
        "cer": total_ed / max(1, total_chars),
        "word_accuracy": correct_words / max(1, len(rows)),
        "predictions": rows,
    }


def save_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "image", "truth", "prediction", "edit_distance"),
        )
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        raise RuntimeError("CUDA is required for the requested 3,000-step run")

    run_dir = args.out_dir / ("smoke" if args.smoke else "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    all_samples = collect_samples(args.data_dir)
    train_samples, test_samples = split_samples(
        all_samples, args.test_size, args.seed, args.fixed_test_manifest
    )
    write_manifest(run_dir / "train_manifest.csv", train_samples)
    write_manifest(run_dir / "test_manifest.csv", test_samples)

    alphabet = sorted(set("".join(item["label"] for item in all_samples)))
    converter = utils.CTCLabelConverter(alphabet)
    train_dataset = WordDataset(
        train_samples,
        args.width,
        args.height,
        not args.no_mirror,
        augment=True,
        rotation_degrees=args.rotation_degrees,
        rotation_probability=args.rotation_probability,
    )
    test_dataset = WordDataset(test_samples, args.width, args.height, not args.no_mirror, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    model = HTR_VT.create_model(nb_cls=len(alphabet) + 1, img_size=[args.height, args.width]).to(device)
    patch_forward_no_final_logit_norm(model)
    with torch.no_grad():
        model.head.bias.zero_()
        model.head.bias[0] = -2.0
    criterion = torch.nn.CTCLoss(reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    configuration = vars(args).copy()
    configuration.update(
        {
            "data_dir": str(args.data_dir),
            "out_dir": str(args.out_dir),
            "fixed_test_manifest": str(args.fixed_test_manifest.resolve()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "test_by_dataset": dict(Counter(item["dataset"] for item in test_samples)),
            "alphabet": alphabet,
            "mirror": not args.no_mirror,
            "final_logit_norm_removed": True,
        }
    )
    (run_dir / "config.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(configuration, ensure_ascii=False), flush=True)

    fields = ("step", "train_loss", "test_loss", "test_cer", "test_word_accuracy", "lr", "elapsed_seconds")
    metrics_path = run_dir / "metrics.csv"
    metrics_handle = metrics_path.open("w", newline="", encoding="utf-8-sig")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=fields)
    metrics_writer.writeheader()
    metrics_handle.flush()

    best_cer = float("inf")
    best_accuracy = -1.0
    best_step = 0
    steps = 1 if args.smoke else args.steps
    iterator = iter(train_loader)
    start_time = time.time()

    try:
        model.train()
        for step in range(1, steps + 1):
            try:
                images, labels, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, labels, _, _ = next(iterator)

            optimizer.zero_grad(set_to_none=True)
            loss, _ = ctc_loss(model, images, labels, converter, criterion, device, amp_enabled)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            if step % args.eval_every == 0 or step == steps:
                result = evaluate(model, test_loader, converter, criterion, device, amp_enabled)
                elapsed = time.time() - start_time
                row = {
                    "step": step,
                    "train_loss": f"{float(loss.detach()):.8f}",
                    "test_loss": f"{result['loss']:.8f}",
                    "test_cer": f"{result['cer']:.8f}",
                    "test_word_accuracy": f"{result['word_accuracy']:.8f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.10f}",
                    "elapsed_seconds": f"{elapsed:.2f}",
                }
                metrics_writer.writerow(row)
                metrics_handle.flush()
                improved = (result["cer"] < best_cer) or (
                    result["cer"] == best_cer and result["word_accuracy"] > best_accuracy
                )
                if improved:
                    best_cer = result["cer"]
                    best_accuracy = result["word_accuracy"]
                    best_step = step
                    atomic_torch_save(
                        {
                            "model": model.state_dict(),
                            "alphabet": alphabet,
                            "step": step,
                            "test_cer": best_cer,
                            "test_word_accuracy": best_accuracy,
                            "config": configuration,
                        },
                        run_dir / "best_model.pth",
                    )
                    save_predictions(run_dir / "best_predictions.csv", result["predictions"])
                print(
                    f"step={step}/{steps} train_loss={float(loss.detach()):.5f} "
                    f"test_loss={result['loss']:.5f} test_CER={result['cer']:.5f} "
                    f"test_word_acc={result['word_accuracy']:.3f} best_step={best_step} "
                    f"best_CER={best_cer:.5f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        metrics_handle.close()

    atomic_torch_save(
        {
            "model": model.state_dict(),
            "alphabet": alphabet,
            "step": steps,
            "best_step": best_step,
            "best_test_cer": best_cer,
            "best_test_word_accuracy": best_accuracy,
            "config": configuration,
        },
        run_dir / "last_model.pth",
    )
    summary = {
        "completed_steps": steps,
        "best_step": best_step,
        "best_test_cer": best_cer,
        "best_test_word_accuracy": best_accuracy,
        "elapsed_seconds": time.time() - start_time,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
