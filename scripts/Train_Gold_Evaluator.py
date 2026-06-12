from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cl.model import SimpleCNN  # noqa: E402


IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"
GOLD_DATA_DIR = IMAGE_DATA_DIR / "Gold_Data"
EVALUATORS_ROOT = REPO_ROOT / "Evaluators"
DEFAULT_LABELS_FILENAME = "labels.csv"
DEFAULT_IMAGE_SIZES = {
    "STL": 96,
    "Flower_102": 224,
}


class AddGaussianNoise:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.sigma
        return (tensor + noise).clamp(0.0, 1.0)


class CsvImageDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        labels_filename: str,
        transform: transforms.Compose,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.labels_path = dataset_dir / labels_filename
        self.transform = transform
        self.rows = read_labels(self.labels_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = self.dataset_dir / row["relative_path"]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(row["label"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen gold evaluator checkpoint from Gold_Data."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. STL.")
    parser.add_argument(
        "--gold-dir",
        type=Path,
        help="Gold dataset directory. Defaults to Image_Data/Gold_Data/<dataset>.",
    )
    parser.add_argument("--labels-filename", default=DEFAULT_LABELS_FILENAME)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of Gold_Data held out for evaluator validation.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--selection-metric",
        choices=("val-accuracy", "val-loss"),
        default="val-accuracy",
        help="Gold evaluator checkpoint selection metric.",
    )
    parser.add_argument(
        "--train-augmentation",
        choices=("none", "standard", "feature-noise"),
        default="none",
        help=(
            "Augmentation used only for gold evaluator training. feature-noise "
            "adds mild blur/color/noise transforms to make the evaluator less "
            "brittle when scoring noisy feature data."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            "Stop evaluator training after this many epochs without validation "
            "metric improvement. Default 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation metric improvement required to reset early stopping.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-size",
        type=int,
        help="Square input size. Defaults to 96 for STL and 224 for Flower_102.",
    )
    parser.add_argument(
        "--evaluators-root",
        type=Path,
        default=EVALUATORS_ROOT,
        help="Root directory for saved frozen evaluator checkpoints.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device, e.g. cuda or cpu.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be in (0, 0.5).")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if args.grad_clip is not None and args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive.")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience cannot be negative.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta cannot be negative.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be positive.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_labels(labels_path: Path) -> list[dict[str, str]]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    with labels_path.open(newline="") as labels_file:
        rows = list(csv.DictReader(labels_file))

    if not rows:
        raise ValueError(f"No rows found in {labels_path}")

    required_columns = {"relative_path", "label", "class_name"}
    missing = required_columns - set(rows[0])
    if missing:
        raise ValueError(f"{labels_path} is missing columns: {sorted(missing)}")

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_gold_dir(dataset: str, gold_dir: Path | None) -> Path:
    if gold_dir is not None:
        return resolve_repo_path(gold_dir)
    return GOLD_DATA_DIR / dataset


def resolve_image_size(dataset: str, image_size: int | None) -> int:
    if image_size is not None:
        return image_size
    if dataset not in DEFAULT_IMAGE_SIZES:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Pass --image-size explicitly."
        )
    return DEFAULT_IMAGE_SIZES[dataset]


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def build_train_transform(image_size: int, augmentation: str) -> transforms.Compose:
    if augmentation == "none":
        return build_eval_transform(image_size)

    if augmentation == "standard":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    if augmentation == "feature-noise":
        blur_kernel = max(3, int(round(image_size * 0.06)) | 1)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 1.5))],
                    p=0.35,
                ),
                transforms.ToTensor(),
                transforms.RandomApply([AddGaussianNoise(sigma=0.03)], p=0.25),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    raise ValueError(f"Unknown train augmentation: {augmentation}")


def infer_num_classes(rows: list[dict[str, str]]) -> int:
    labels = {int(row["label"]) for row in rows}
    if not labels:
        raise ValueError("Cannot infer num_classes from empty labels.")
    return max(labels) + 1


def make_loader(
    dataset_dir: Path,
    labels_filename: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> tuple[DataLoader, list[dict[str, str]]]:
    dataset = CsvImageDataset(
        dataset_dir=dataset_dir,
        labels_filename=labels_filename,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.rows


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def stratified_gold_split(
    rows: list[dict[str, str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_label[int(row["label"])].append(index)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for label_indices in indices_by_label.values():
        shuffled = label_indices[:]
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_fraction))
        val_count = min(val_count, len(shuffled) - 1) if len(shuffled) > 1 else 0
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def rows_for_indices(
    rows: list[dict[str, str]],
    indices: list[int],
) -> list[dict[str, str]]:
    return [rows[index] for index in indices]


def build_run_dir(evaluators_root: Path, dataset: str) -> Path:
    return resolve_repo_path(evaluators_root) / dataset


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"Gold evaluator epoch {epoch}", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size
        progress.set_postfix(
            loss=total_loss / total_examples,
            acc=total_correct / total_examples,
        )

    return total_loss / total_examples, total_correct / total_examples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    per_class_correct = [0 for _ in range(num_classes)]
    per_class_total = [0 for _ in range(num_classes)]

    for images, labels in tqdm(loader, desc="Evaluate gold evaluator", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

        for label, prediction in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
            per_class_total[label] += 1
            per_class_correct[label] += int(label == prediction)

    per_class_rows = []
    for label in range(num_classes):
        total = per_class_total[label]
        correct = per_class_correct[label]
        per_class_rows.append(
            {
                "label": label,
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else None,
            }
        )

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        per_class_rows,
    )


def save_outputs(
    run_dir: Path,
    model: nn.Module,
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    per_class_rows: list[dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_csv(run_dir / "history.csv", history)
    write_csv(run_dir / "per_class_accuracy.csv", per_class_rows)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    gold_dir = resolve_gold_dir(args.dataset, args.gold_dir)
    image_size = resolve_image_size(args.dataset, args.image_size)
    device = torch.device(args.device)
    train_transform = build_train_transform(image_size, args.train_augmentation)
    eval_transform = build_eval_transform(image_size)

    gold_train_dataset = CsvImageDataset(
        dataset_dir=gold_dir,
        labels_filename=args.labels_filename,
        transform=train_transform,
    )
    gold_eval_dataset = CsvImageDataset(
        dataset_dir=gold_dir,
        labels_filename=args.labels_filename,
        transform=eval_transform,
    )
    gold_rows = gold_eval_dataset.rows
    gold_train_indices, gold_val_indices = stratified_gold_split(
        rows=gold_rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    gold_train_rows = rows_for_indices(gold_rows, gold_train_indices)
    gold_val_rows = rows_for_indices(gold_rows, gold_val_indices)
    train_loader = make_dataloader(
        dataset=Subset(gold_train_dataset, gold_train_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    eval_loader = make_dataloader(
        dataset=Subset(gold_eval_dataset, gold_val_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    num_classes = infer_num_classes(gold_rows)
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_dir = build_run_dir(args.evaluators_root, args.dataset)
    history = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_per_class_rows = []
    last_per_class_rows = []
    epochs_without_improvement = 0
    stopped_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
        )
        eval_loss, eval_accuracy, per_class_rows = evaluate(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "gold_validation_loss": eval_loss,
            "gold_validation_accuracy": eval_accuracy,
        }
        history.append(row)
        last_per_class_rows = per_class_rows
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"gold_val_loss={eval_loss:.4f} gold_val_acc={eval_accuracy:.4f}"
        )

        if args.selection_metric == "val-accuracy":
            metric_improved = eval_accuracy > best_accuracy + args.early_stop_min_delta
        else:
            metric_improved = eval_loss < best_loss - args.early_stop_min_delta

        if metric_improved:
            best_accuracy = eval_accuracy
            best_loss = eval_loss
            best_epoch = epoch
            best_per_class_rows = per_class_rows
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            stopped_epoch = epoch
            print(
                f"Early stopping gold evaluator at epoch={epoch}; "
                f"selection_metric={args.selection_metric} "
                f"best_epoch={best_epoch} "
                f"best_val_loss={best_loss:.4f} "
                f"best_val_acc={best_accuracy:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "method": "train_gold_evaluator",
        "dataset": args.dataset,
        "num_classes": num_classes,
        "gold_dir": str(gold_dir),
        "labels_file": str(gold_dir / args.labels_filename),
        "gold_examples": len(gold_rows),
        "gold_train_examples": len(gold_train_rows),
        "gold_validation_examples": len(gold_val_rows),
        "validation_fraction": args.val_fraction,
        "validation_source": "gold_data_stratified_split",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "selection_metric": args.selection_metric,
        "train_augmentation": args.train_augmentation,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stopped": stopped_epoch > 0,
        "stopped_epoch": stopped_epoch,
        "seed": args.seed,
        "device": str(device),
        "checkpoint_selection": f"best_gold_validation_{args.selection_metric}",
        "best_epoch": best_epoch,
        "selected_gold_validation_loss": best_loss,
        "best_gold_validation_accuracy": best_accuracy,
        "final_gold_validation_accuracy": history[-1]["gold_validation_accuracy"],
        "checkpoint": str(run_dir / "model.pt"),
    }
    save_outputs(
        run_dir=run_dir,
        model=model,
        history=history,
        metrics=metrics,
        per_class_rows=best_per_class_rows or last_per_class_rows,
    )
    print(f"Saved frozen gold evaluator checkpoint to {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
