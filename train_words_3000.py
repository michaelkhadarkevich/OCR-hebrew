import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from types import MethodType

import editdistance
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter

from data import transform as paper_transform
from model import HTR_VT
from utils import sam as sam_utils
from utils import utils


DATASETS = (
    "arielOnlyWord",
    "agadaOnlyWord",
    "RockOnlyWord",
    "phisicsTigulOnlyWords",
)

TRAINING_RECIPE_VERSION = "global-cosine-bound-ema-v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HTR-VT on Hebrew word, line, or combined datasets"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/hebrew_words_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=int, default=90)
    parser.add_argument(
        "--fixed-test-manifest",
        type=Path,
        default=Path("data/manifests/fixed_hebrew_test_words.csv"),
        help="Persistent test split reused across runs; created once when missing",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adamw", "sam"), default="adamw")
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--warm-up-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    parser.add_argument("--max-span-length", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rotation-degrees", type=float, default=0.0)
    parser.add_argument("--rotation-probability", type=float, default=0.5)
    parser.add_argument("--stretch-factor", type=float, default=0.0)
    parser.add_argument("--blur-probability", type=float, default=0.0)
    parser.add_argument("--noise-probability", type=float, default=0.0)
    parser.add_argument(
        "--paper-augmentation",
        action="store_true",
        help="Use the paper's projective, erosion/dilation, color-jitter and elastic transforms",
    )
    parser.add_argument("--paper-augmentation-probability", type=float, default=0.5)
    parser.add_argument("--projective-value", type=float, default=8.0)
    parser.add_argument("--dila-ero-max-kernel", type=int, default=3)
    parser.add_argument("--elastic-grid-width", type=int, default=8)
    parser.add_argument("--elastic-grid-height", type=int, default=4)
    parser.add_argument("--elastic-magnitude", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument(
        "--numerical-diagnostics",
        action="store_true",
        help="Log per-step loss/gradient/AMP diagnostics and stop at the first non-finite loss",
    )
    parser.add_argument(
        "--chunk-epochs",
        type=int,
        default=0,
        help="Stop cleanly after this many data-loader epochs; zero runs through --steps",
    )
    parser.add_argument("--train-dirs", type=Path, nargs="+", help="Explicit training folders")
    parser.add_argument("--test-dir", type=Path, help="Explicit validation folder used to select checkpoints")
    parser.add_argument("--final-test-dir", type=Path, help="Untouched final-test folder evaluated after training")
    parser.add_argument("--validation-dir", type=Path, help="Folder to sample validation rows from")
    parser.add_argument("--validation-size", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Evaluations without improvement")
    parser.add_argument("--lr-plateau-patience", type=int, default=0, help="Evaluations before reducing LR")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--initial-best-checkpoint", type=Path)
    parser.add_argument(
        "--strip-whitespace",
        action="store_true",
        help="Remove whitespace from labels (use when spaces are not scored)",
    )
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


def collect_samples_from_dirs(directories, strip_whitespace=False):
    samples = []
    for root in directories:
        for image_path in sorted(root.rglob("*.png")):
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                continue
            label = label_path.read_text(encoding="utf-8-sig").strip()
            if strip_whitespace:
                label = "".join(label.split())
            if not label:
                continue
            samples.append(
                {
                    "dataset": root.name,
                    "image": str(image_path.resolve()),
                    "label_file": str(label_path.resolve()),
                    "label": label,
                }
            )
    if not samples:
        raise RuntimeError("No paired PNG/TXT samples were found in the explicit folders")
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
        stretch_factor=0.0,
        blur_probability=0.0,
        noise_probability=0.0,
        paper_augmentation=False,
        paper_augmentation_probability=0.5,
        projective_value=8.0,
        dila_ero_max_kernel=3,
        elastic_grid_width=8,
        elastic_grid_height=4,
        elastic_magnitude=2,
    ):
        self.samples = samples
        self.width = width
        self.height = height
        self.mirror = mirror
        self.augment = augment
        self.rotation_degrees = rotation_degrees
        self.rotation_probability = rotation_probability
        self.stretch_factor = stretch_factor
        self.blur_probability = blur_probability
        self.noise_probability = noise_probability
        self.paper_augmentation = paper_augmentation
        self.paper_augmentation_probability = paper_augmentation_probability
        self.projective_value = projective_value
        self.dila_ero_max_kernel = dila_ero_max_kernel
        self.elastic_grid_width = elastic_grid_width
        self.elastic_grid_height = elastic_grid_height
        self.elastic_magnitude = elastic_magnitude

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
        if self.augment and self.paper_augmentation:
            probability = self.paper_augmentation_probability
            if random.random() < probability:
                image = paper_transform.RandomTransform(self.projective_value)(image)
            if random.random() < probability:
                kernel = (
                    random.randint(1, self.dila_ero_max_kernel),
                    random.randint(1, self.dila_ero_max_kernel),
                )
                operation = paper_transform.Erosion if random.random() < 0.5 else paper_transform.Dilation
                image = operation(kernel, 1)(image)
            if random.random() < probability:
                image = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)(image)
            if random.random() < probability:
                grid_width = max(2, min(self.elastic_grid_width, image.width // 4))
                grid_height = max(2, min(self.elastic_grid_height, image.height // 4))
                image = paper_transform.ElasticDistortion(
                    grid=(grid_width, grid_height),
                    magnitude=(self.elastic_magnitude, self.elastic_magnitude),
                    min_sep=(1, 1),
                )(image)
        elif self.augment:
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.25:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            if self.stretch_factor > 0:
                scale = random.uniform(1.0 - self.stretch_factor, 1.0 + self.stretch_factor)
                image = image.resize((max(1, round(image.width * scale)), image.height), Image.Resampling.BILINEAR)
            if random.random() < self.blur_probability:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))
        new_width = min(self.width, max(1, round(image.width * self.height / image.height)))
        image = image.resize((new_width, self.height), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (self.width, self.height), 255)
        canvas.paste(image, (0, 0))
        pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).clone()
        tensor = pixels.reshape(self.height, self.width).unsqueeze(0).float().div_(255.0)
        if self.augment and random.random() < self.noise_probability:
            tensor = (tensor + torch.randn_like(tensor) * random.uniform(0.005, 0.02)).clamp_(0.0, 1.0)
        return tensor, item["label"], item["dataset"], item["image"]


def collate(batch):
    images, labels, datasets, paths = zip(*batch)
    return torch.stack(images), list(labels), list(datasets), list(paths)


def forward_no_final_logit_norm(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
    x = self.layer_norm(x)
    x = self.patch_embed(x)
    batch, channels, _, _ = x.shape
    x = x.view(batch, channels, -1).permute(0, 2, 1)
    if use_masking:
        x = self.random_masking(x, mask_ratio, max_span_length)
    x = x + self.pos_embed
    for block in self.blocks:
        x = block(x)
    return self.head(self.norm(x))


def patch_forward_no_final_logit_norm(model):
    # A closure over model survives deepcopy unchanged and makes an EMA copy
    # execute the raw model. A bound method is rebound to the copied instance.
    model.forward = MethodType(forward_no_final_logit_norm, model)


def ctc_loss(
    model,
    images,
    labels,
    converter,
    criterion,
    device,
    amp_enabled,
    mask_ratio=0.0,
    max_span_length=1,
    use_masking=False,
):
    images = images.to(device, non_blocking=True)
    targets, target_lengths = converter.encode(labels)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        logits = model(
            images,
            mask_ratio=mask_ratio,
            max_span_length=max_span_length,
            use_masking=use_masking,
        )
        time_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.int32, device=device)
        log_probs = logits.float().permute(1, 0, 2).log_softmax(2)
        loss = criterion(log_probs, targets, time_lengths, target_lengths)
    return loss, logits


@torch.inference_mode()
def evaluate(model, loader, converter, criterion, device, amp_enabled):
    was_training = model.training
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
    if was_training:
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


def optimizer_state_dict(optimizer, optimizer_name):
    if optimizer_name == "sam":
        return optimizer.base_optimizer.state_dict()
    return optimizer.state_dict()


def load_optimizer_state_dict(optimizer, optimizer_name, state_dict):
    if optimizer_name == "sam":
        optimizer.base_optimizer.load_state_dict(state_dict)
        optimizer.param_groups = optimizer.base_optimizer.param_groups
    else:
        optimizer.load_state_dict(state_dict)


def training_checkpoint_payload(
    model,
    model_ema,
    optimizer,
    optimizer_name,
    scaler,
    data_generator,
    alphabet,
    step,
    best_step,
    best_cer,
    best_accuracy,
    configuration,
    elapsed_seconds,
    rolling_train_loss,
    rolling_train_batches,
):
    return {
        "model": model.state_dict(),
        "ema_model": model_ema.ema.state_dict() if model_ema is not None else None,
        "optimizer": optimizer_state_dict(optimizer, optimizer_name),
        "scaler": scaler.state_dict(),
        "data_generator_state": data_generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "alphabet": alphabet,
        "step": step,
        "best_step": best_step,
        "best_test_cer": best_cer,
        "best_test_word_accuracy": best_accuracy,
        "elapsed_seconds": elapsed_seconds,
        "rolling_train_loss": rolling_train_loss,
        "rolling_train_batches": rolling_train_batches,
        "config": configuration,
    }


def cosine_learning_rate(step, warm_up_steps, total_steps, max_lr, min_lr):
    if warm_up_steps > 0 and step <= warm_up_steps:
        return max_lr * step / warm_up_steps
    decay_steps = max(1, total_steps - warm_up_steps)
    progress = min(1.0, max(0.0, (step - warm_up_steps) / decay_steps))
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        raise RuntimeError("CUDA is required for the requested 3,000-step run")

    run_dir = args.out_dir / ("smoke" if args.smoke else "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    final_test_samples = None
    if args.train_dirs or args.test_dir:
        if not args.train_dirs or not args.test_dir:
            raise ValueError("--train-dirs and --test-dir must be provided together")
        train_samples = collect_samples_from_dirs(args.train_dirs, args.strip_whitespace)
        validation_samples = collect_samples_from_dirs([args.test_dir], args.strip_whitespace)
        final_test_samples = (
            collect_samples_from_dirs([args.final_test_dir], args.strip_whitespace)
            if args.final_test_dir
            else None
        )
        held_out_samples = validation_samples + (final_test_samples or [])
        overlap = {item["image"] for item in train_samples} & {item["image"] for item in held_out_samples}
        if overlap:
            raise RuntimeError(f"Explicit train/test folders overlap; first: {next(iter(overlap))}")
        if args.final_test_dir:
            test_samples = validation_samples
        elif args.validation_size:
            if not args.validation_dir:
                raise ValueError("--validation-dir is required when --validation-size is used")
            candidates = collect_samples_from_dirs([args.validation_dir], args.strip_whitespace)
            if args.validation_size >= len(candidates):
                raise ValueError("Validation size must be smaller than its source folder")
            validation_rng = random.Random(args.seed)
            test_samples = validation_rng.sample(candidates, args.validation_size)
            validation_paths = {item["image"] for item in test_samples}
            train_samples = [item for item in train_samples if item["image"] not in validation_paths]
        else:
            test_samples = validation_samples
            final_test_samples = None
        all_samples = train_samples + test_samples + (final_test_samples or [])
    else:
        all_samples = collect_samples(args.data_dir)
        train_samples, test_samples = split_samples(
            all_samples, args.test_size, args.seed, args.fixed_test_manifest
        )
    write_manifest(run_dir / "train_manifest.csv", train_samples)
    write_manifest(run_dir / "validation_manifest.csv", test_samples)
    if final_test_samples is not None:
        write_manifest(run_dir / "test_manifest.csv", final_test_samples)
    else:
        write_manifest(run_dir / "test_manifest.csv", test_samples)

    alphabet = sorted(set("".join(item["label"] for item in train_samples + test_samples)))
    if final_test_samples:
        unseen_final_characters = sorted(
            set("".join(item["label"] for item in final_test_samples)) - set(alphabet)
        )
        if unseen_final_characters:
            raise RuntimeError(
                "Final test contains characters absent from train/validation: "
                + repr("".join(unseen_final_characters))
            )
    converter = utils.CTCLabelConverter(alphabet)
    train_dataset = WordDataset(
        train_samples,
        args.width,
        args.height,
        not args.no_mirror,
        augment=True,
        rotation_degrees=args.rotation_degrees,
        rotation_probability=args.rotation_probability,
        stretch_factor=args.stretch_factor,
        blur_probability=args.blur_probability,
        noise_probability=args.noise_probability,
        paper_augmentation=args.paper_augmentation,
        paper_augmentation_probability=args.paper_augmentation_probability,
        projective_value=args.projective_value,
        dila_ero_max_kernel=args.dila_ero_max_kernel,
        elastic_grid_width=args.elastic_grid_width,
        elastic_grid_height=args.elastic_grid_height,
        elastic_magnitude=args.elastic_magnitude,
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
    final_test_loader = None
    if final_test_samples is not None:
        final_test_dataset = WordDataset(
            final_test_samples, args.width, args.height, not args.no_mirror, augment=False
        )
        final_test_loader = DataLoader(
            final_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )

    model = HTR_VT.create_model(nb_cls=len(alphabet) + 1, img_size=[args.height, args.width]).to(device)
    patch_forward_no_final_logit_norm(model)
    resume_checkpoint = None
    start_step = 0
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if (args.ema_decay > 0 or args.warm_up_steps > 0) and (
            resume_checkpoint.get("config", {}).get("training_recipe_version") != TRAINING_RECIPE_VERSION
        ):
            raise RuntimeError(
                "This checkpoint predates the verified EMA/global-cosine recipe. "
                "Start a fresh run in a new output directory; resuming it cannot repair historical training."
            )
        if resume_checkpoint["alphabet"] != alphabet:
            raise RuntimeError("Resume checkpoint alphabet does not match the current datasets")
        model.load_state_dict(resume_checkpoint["model"])
        start_step = int(resume_checkpoint.get("step", 0))
    else:
        with torch.no_grad():
            model.head.bias.zero_()
            model.head.bias[0] = -2.0
    criterion = torch.nn.CTCLoss(reduction="mean", zero_infinity=True)
    if args.optimizer == "sam":
        optimizer = sam_utils.SAM(
            model.parameters(),
            torch.optim.AdamW,
            rho=args.sam_rho,
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_plateau_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_plateau_patience
        )
    # The released paper training loop uses SAM in full precision. Keeping AMP for
    # ordinary AdamW is useful, but scaled two-pass SAM gradients can become
    # non-finite before the perturbation step.
    amp_enabled = device.type == "cuda" and args.optimizer != "sam"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model_ema = utils.ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None
    if model_ema is not None:
        model_ema.ema.eval()
    if resume_checkpoint is not None:
        if "optimizer" not in resume_checkpoint:
            raise RuntimeError(
                "Resume checkpoint predates full-state checkpoints; start this staged run from step 0"
            )
        load_optimizer_state_dict(
            optimizer, args.optimizer, resume_checkpoint["optimizer"]
        )
        if resume_checkpoint.get("scaler"):
            scaler.load_state_dict(resume_checkpoint["scaler"])
        if model_ema is not None and resume_checkpoint.get("ema_model") is not None:
            model_ema.ema.load_state_dict(resume_checkpoint["ema_model"])
        if resume_checkpoint.get("data_generator_state") is not None:
            generator.set_state(resume_checkpoint["data_generator_state"].cpu())
        if resume_checkpoint.get("python_rng_state") is not None:
            random.setstate(resume_checkpoint["python_rng_state"])
        if resume_checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(resume_checkpoint["numpy_rng_state"])
        if resume_checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and resume_checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_checkpoint["cuda_rng_state_all"]]
            )

    configuration = vars(args).copy()
    configuration.update(
        {
            "training_recipe_version": TRAINING_RECIPE_VERSION,
            "data_dir": str(args.data_dir),
            "out_dir": str(args.out_dir),
            "fixed_test_manifest": str(args.fixed_test_manifest.resolve()),
            "train_dirs": [str(path.resolve()) for path in args.train_dirs] if args.train_dirs else None,
            "test_dir": str(args.test_dir.resolve()) if args.test_dir else None,
            "final_test_dir": str(args.final_test_dir.resolve()) if args.final_test_dir else None,
            "validation_dir": str(args.validation_dir.resolve()) if args.validation_dir else None,
            "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
            "initial_best_checkpoint": str(args.initial_best_checkpoint.resolve()) if args.initial_best_checkpoint else None,
            "start_step": start_step,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "test_by_dataset": dict(Counter(item["dataset"] for item in test_samples)),
            "final_test_samples": len(final_test_samples) if final_test_samples is not None else None,
            "alphabet": alphabet,
            "mirror": not args.no_mirror,
            "final_logit_norm_removed": True,
        }
    )
    (run_dir / "config.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(configuration, ensure_ascii=False), flush=True)

    fields = ("step", "train_loss", "test_loss", "test_cer", "test_word_accuracy", "lr", "elapsed_seconds")
    metrics_path = run_dir / "metrics.csv"
    preserved_metric_rows = []
    if resume_checkpoint is not None and metrics_path.exists():
        with metrics_path.open("r", newline="", encoding="utf-8-sig") as existing_metrics:
            preserved_metric_rows = [
                row
                for row in csv.DictReader(existing_metrics)
                if int(row["step"]) <= start_step
            ]
    metrics_handle = metrics_path.open("w", newline="", encoding="utf-8-sig")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=fields)
    metrics_writer.writeheader()
    metrics_writer.writerows(preserved_metric_rows)
    metrics_handle.flush()
    diagnostics_handle = None
    diagnostics_writer = None
    if args.numerical_diagnostics:
        diagnostics_handle = (run_dir / "numerical_diagnostics.csv").open(
            "w", newline="", encoding="utf-8-sig"
        )
        diagnostics_writer = csv.DictWriter(
            diagnostics_handle,
            fieldnames=(
                "step",
                "loss",
                "gradient_norm",
                "gradient_finite",
                "amp_scale",
                "lr",
                "datasets",
                "paths",
            ),
        )
        diagnostics_writer.writeheader()
        diagnostics_handle.flush()

    best_cer = float("inf")
    best_accuracy = -1.0
    best_step = 0
    initial_best_path = args.initial_best_checkpoint
    if initial_best_path is None and resume_checkpoint is not None:
        existing_best_path = run_dir / "best_model.pth"
        if existing_best_path.exists():
            initial_best_path = existing_best_path
    if initial_best_path:
        initial_best = torch.load(initial_best_path, map_location="cpu", weights_only=False)
        if initial_best["alphabet"] != alphabet:
            raise RuntimeError("Initial best checkpoint alphabet does not match the current datasets")
        best_cer = float(initial_best.get("test_cer", float("inf")))
        best_accuracy = float(initial_best.get("test_word_accuracy", -1.0))
        best_step = int(initial_best.get("step", 0))
        destination_best_path = run_dir / "best_model.pth"
        if initial_best_path.resolve() != destination_best_path.resolve():
            shutil.copy2(initial_best_path, destination_best_path)
        previous_predictions = initial_best_path.with_name("best_predictions.csv")
        if previous_predictions.exists():
            destination_predictions = run_dir / "best_predictions.csv"
            if previous_predictions.resolve() != destination_predictions.resolve():
                shutil.copy2(previous_predictions, destination_predictions)
    evaluations_without_improvement = 0
    completed_steps = 0
    final_step = start_step + 1 if args.smoke else args.steps
    if not args.smoke and args.chunk_epochs > 0:
        final_step = min(args.steps, start_step + args.chunk_epochs * len(train_loader))
    if final_step <= start_step:
        raise ValueError("--steps must be greater than the resume checkpoint step")
    iterator = iter(train_loader)
    elapsed_before_resume = (
        float(resume_checkpoint.get("elapsed_seconds", 0.0))
        if resume_checkpoint is not None
        else 0.0
    )
    start_time = time.time()
    rolling_train_loss = (
        float(resume_checkpoint.get("rolling_train_loss", 0.0))
        if resume_checkpoint is not None
        else 0.0
    )
    rolling_train_batches = (
        int(resume_checkpoint.get("rolling_train_batches", 0))
        if resume_checkpoint is not None
        else 0
    )

    try:
        model.train()
        for step in range(start_step + 1, final_step + 1):
            completed_steps = step
            try:
                images, labels, batch_datasets, batch_paths = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, labels, batch_datasets, batch_paths = next(iterator)

            if args.warm_up_steps > 0:
                current_lr = cosine_learning_rate(
                    step, args.warm_up_steps, args.steps, args.lr, args.min_lr
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            loss, _ = ctc_loss(
                model,
                images,
                labels,
                converter,
                criterion,
                device,
                amp_enabled,
                mask_ratio=args.mask_ratio,
                max_span_length=args.max_span_length,
                use_masking=args.mask_ratio > 0,
            )
            if args.numerical_diagnostics and not torch.isfinite(loss):
                failure = {
                    "step": step,
                    "stage": "forward_loss",
                    "loss": str(float(loss.detach())),
                    "amp_scale": float(scaler.get_scale()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "datasets": batch_datasets,
                    "paths": batch_paths,
                    "labels": labels,
                }
                (run_dir / "numerical_failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise RuntimeError(f"Non-finite training loss at step {step}")
            if args.optimizer == "sam":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                first_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(first_grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    raise RuntimeError(f"Non-finite SAM gradient at step {step}")
                optimizer.first_step(zero_grad=True)
                second_loss, _ = ctc_loss(
                    model,
                    images,
                    labels,
                    converter,
                    criterion,
                    device,
                    amp_enabled,
                    mask_ratio=args.mask_ratio,
                    max_span_length=args.max_span_length,
                    use_masking=args.mask_ratio > 0,
                )
                scaler.scale(second_loss).backward()
                optimizer.restore_weights()
                scaler.unscale_(optimizer.base_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer.base_optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if diagnostics_writer is not None:
                    diagnostics_writer.writerow(
                        {
                            "step": step,
                            "loss": f"{float(loss.detach()):.10g}",
                            "gradient_norm": f"{float(gradient_norm):.10g}",
                            "gradient_finite": bool(torch.isfinite(gradient_norm)),
                            "amp_scale": f"{float(scaler.get_scale()):.10g}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.10g}",
                            "datasets": "|".join(batch_datasets),
                            "paths": "|".join(batch_paths),
                        }
                    )
                    if step % 100 == 0:
                        diagnostics_handle.flush()
                scaler.step(optimizer)
                scaler.update()

            if model_ema is not None:
                model_ema.update(model, num_updates=step / 2)
            rolling_train_loss += float(loss.detach())
            rolling_train_batches += 1

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                atomic_torch_save(
                    training_checkpoint_payload(
                        model,
                        model_ema,
                        optimizer,
                        args.optimizer,
                        scaler,
                        generator,
                        alphabet,
                        step,
                        best_step,
                        best_cer,
                        best_accuracy,
                        configuration,
                        elapsed_before_resume + time.time() - start_time,
                        rolling_train_loss,
                        rolling_train_batches,
                    ),
                    run_dir / "last_model.pth",
                )

            if step % args.eval_every == 0 or step == args.steps or args.smoke:
                evaluation_model = model_ema.ema if model_ema is not None else model
                result = evaluate(
                    evaluation_model, test_loader, converter, criterion, device, amp_enabled
                )
                elapsed = elapsed_before_resume + time.time() - start_time
                mean_train_loss = rolling_train_loss / max(1, rolling_train_batches)
                row = {
                    "step": step,
                    "train_loss": f"{mean_train_loss:.8f}",
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
                            "model": evaluation_model.state_dict(),
                            "alphabet": alphabet,
                            "step": step,
                            "test_cer": best_cer,
                            "test_word_accuracy": best_accuracy,
                            "config": configuration,
                        },
                        run_dir / "best_model.pth",
                    )
                    save_predictions(run_dir / "best_predictions.csv", result["predictions"])
                    evaluations_without_improvement = 0
                else:
                    evaluations_without_improvement += 1
                if scheduler is not None:
                    scheduler.step(result["cer"])
                print(
                    f"step={step}/{final_step} train_loss={mean_train_loss:.5f} "
                    f"test_loss={result['loss']:.5f} test_CER={result['cer']:.5f} "
                    f"test_word_acc={result['word_accuracy']:.3f} best_step={best_step} "
                    f"best_CER={best_cer:.5f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                rolling_train_loss = 0.0
                rolling_train_batches = 0
                if (
                    args.early_stopping_patience > 0
                    and evaluations_without_improvement >= args.early_stopping_patience
                ):
                    print(
                        f"early_stopping step={step} best_step={best_step} "
                        f"evaluations_without_improvement={evaluations_without_improvement}",
                        flush=True,
                    )
                    break
    finally:
        metrics_handle.close()
        if diagnostics_handle is not None:
            diagnostics_handle.close()

    atomic_torch_save(
        training_checkpoint_payload(
            model,
            model_ema,
            optimizer,
            args.optimizer,
            scaler,
            generator,
            alphabet,
            completed_steps,
            best_step,
            best_cer,
            best_accuracy,
            configuration,
            elapsed_before_resume + time.time() - start_time,
            rolling_train_loss,
            rolling_train_batches,
        ),
        run_dir / "last_model.pth",
    )
    final_test_result = None
    training_complete = completed_steps >= args.steps
    if training_complete and final_test_loader is not None:
        best_checkpoint = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        final_test_result = evaluate(model, final_test_loader, converter, criterion, device, amp_enabled)
        save_predictions(run_dir / "final_test_predictions.csv", final_test_result["predictions"])

    summary = {
        "training_recipe_version": TRAINING_RECIPE_VERSION,
        "status": "complete" if training_complete else "checkpointed",
        "completed_steps": completed_steps,
        "target_steps": args.steps,
        "best_step": best_step,
        "best_validation_cer": best_cer,
        "best_validation_line_accuracy": best_accuracy,
        "final_test_cer": final_test_result["cer"] if final_test_result else None,
        "final_test_line_accuracy": final_test_result["word_accuracy"] if final_test_result else None,
        "elapsed_seconds": elapsed_before_resume + time.time() - start_time,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
