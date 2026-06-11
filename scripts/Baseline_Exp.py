from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cl.model import SimpleCNN


IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"
TRAIN_CLEAN_DIR = IMAGE_DATA_DIR / "Train_Clean_Data"
TEST_CLEAN_DIR = IMAGE_DATA_DIR / "Test_Clean_Data"
RESULTS_ROOT = REPO_ROOT / "Experiments_Results"
DEFAULT_LABELS_FILENAME = "labels.csv"
WITH_GOLD_LABELS_FILENAME = "labels_before_gold_split.csv"
DEFAULT_IMAGE_SIZES = {
    "STL": 96,
    "Flower_102": 224,
}


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
        description=(
            "Generic baseline experiment: train on one image dataset directory "
            "and evaluate on another. Each directory must contain images/ and "
            "a labels.csv-style file."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. STL or Flower_102.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        help=(
            "Training dataset directory. Defaults to "
            "Image_Data/Train_Clean_Data/<dataset>."
        ),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help=(
            "Testing dataset directory. Defaults to "
            "Image_Data/Test_Clean_Data/<dataset>."
        ),
    )
    parser.add_argument(
        "--train-labels-filename",
        default=None,
        help=(
            "Training labels filename inside --train-dir. Defaults to labels.csv, "
            "or labels_before_gold_split.csv when --include-gold-in-train is used."
        ),
    )
    parser.add_argument(
        "--test-labels-filename",
        default=DEFAULT_LABELS_FILENAME,
        help="Testing labels filename inside --test-dir.",
    )
    parser.add_argument(
        "--include-gold-in-train",
        action="store_true",
        help=(
            "For clean train dirs, train with labels_before_gold_split.csv instead "
            "of labels.csv. This does not read Gold_Data."
        ),
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Optional output leaf folder name. Defaults to a name derived from "
            "the train/test directories."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-size",
        type=int,
        help=(
            "Square input size. Defaults to 96 for STL and 224 for Flower_102. "
            "Use a larger value for Flower_102 if GPU memory allows."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory for experiment outputs.",
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
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if args.include_gold_in_train and args.train_labels_filename is not None:
        raise ValueError(
            "Use either --include-gold-in-train or --train-labels-filename, not both."
        )


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
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_train_dir(dataset: str, train_dir: Path | None) -> Path:
    if train_dir is not None:
        return resolve_repo_path(train_dir)
    return TRAIN_CLEAN_DIR / dataset


def resolve_test_dir(dataset: str, test_dir: Path | None) -> Path:
    if test_dir is not None:
        return resolve_repo_path(test_dir)
    return TEST_CLEAN_DIR / dataset


def resolve_train_labels_filename(args: argparse.Namespace) -> str:
    if args.train_labels_filename is not None:
        return args.train_labels_filename
    if args.include_gold_in_train:
        return WITH_GOLD_LABELS_FILENAME
    return DEFAULT_LABELS_FILENAME


def resolve_image_size(dataset: str, image_size: int | None) -> int:
    if image_size is not None:
        return image_size
    if dataset not in DEFAULT_IMAGE_SIZES:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Pass --image-size explicitly."
        )
    return DEFAULT_IMAGE_SIZES[dataset]


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def infer_num_classes(*row_groups: list[dict[str, str]]) -> int:
    labels = {int(row["label"]) for rows in row_groups for row in rows}
    if not labels:
        raise ValueError("Cannot infer num_classes from empty labels.")
    return max(labels) + 1


def make_dataloaders(
    train_dir: Path,
    test_dir: Path,
    train_labels_filename: str,
    test_labels_filename: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> tuple[DataLoader, DataLoader, list[dict[str, str]], list[dict[str, str]]]:
    transform = build_transform(image_size)
    train_dataset = CsvImageDataset(
        train_dir,
        labels_filename=train_labels_filename,
        transform=transform,
    )
    test_dataset = CsvImageDataset(
        test_dir,
        labels_filename=test_labels_filename,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, train_dataset.rows, test_dataset.rows


def source_group(path: Path, split: str) -> str:
    try:
        relative_parts = path.resolve().relative_to(IMAGE_DATA_DIR.resolve()).parts
    except ValueError:
        return f"{split}_Custom"

    if not relative_parts:
        return f"{split}_Custom"

    source_root = relative_parts[0]
    if source_root == f"{split}_Clean_Data":
        return f"{split}_Clean"
    if source_root == f"{split}_Noise_Data":
        return f"{split}_Noise"
    return f"{split}_Custom"


def relative_result_path(path: Path) -> Path:
    try:
        relative = path.resolve().relative_to(IMAGE_DATA_DIR.resolve())
    except ValueError:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())

    parts = list(relative.parts)
    if len(parts) == 2 and parts[0] in {"Train_Clean_Data", "Test_Clean_Data"}:
        return Path("clean")
    if len(parts) >= 2 and parts[0].endswith("_Data"):
        parts = parts[2:]
    return Path(*parts) if parts else Path(path.name)


def default_run_path(
    train_dir: Path,
    test_dir: Path,
    train_labels_filename: str,
) -> Path:
    train_path = relative_result_path(train_dir)
    test_path = relative_result_path(test_dir)
    if train_labels_filename == WITH_GOLD_LABELS_FILENAME:
        train_path = train_path / "with_gold_labels"
    if test_path == Path("clean") or test_path == train_path:
        return train_path
    return train_path / "test" / test_path


def build_run_dir(
    results_root: Path,
    dataset: str,
    train_dir: Path,
    test_dir: Path,
    train_labels_filename: str,
    run_name: str | None,
) -> Path:
    experiment_group = f"{source_group(train_dir, 'Train')}_{source_group(test_dir, 'Test')}"
    leaf_path = (
        Path(run_name)
        if run_name is not None
        else default_run_path(
            train_dir=train_dir,
            test_dir=test_dir,
            train_labels_filename=train_labels_filename,
        )
    )
    return resolve_repo_path(results_root) / experiment_group / "Baseline_Exp" / dataset / leaf_path


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
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

    for images, labels in tqdm(loader, desc="Evaluate", leave=False):
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

    train_dir = resolve_train_dir(args.dataset, args.train_dir)
    test_dir = resolve_test_dir(args.dataset, args.test_dir)
    train_labels_filename = resolve_train_labels_filename(args)
    image_size = resolve_image_size(args.dataset, args.image_size)

    train_loader, test_loader, train_rows, test_rows = make_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_labels_filename=train_labels_filename,
        test_labels_filename=args.test_labels_filename,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )

    num_classes = infer_num_classes(train_rows, test_rows)
    device = torch.device(args.device)
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_dir = build_run_dir(
        results_root=args.results_root,
        dataset=args.dataset,
        train_dir=train_dir,
        test_dir=test_dir,
        train_labels_filename=train_labels_filename,
        run_name=args.run_name,
    )
    history = []
    best_test_accuracy = -1.0
    best_state = None
    best_epoch = 0
    last_per_class_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        test_loss, test_accuracy, per_class_rows = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        history.append(row)
        last_per_class_rows = per_class_rows
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}"
        )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "train_labels_file": str(train_dir / train_labels_filename),
        "test_labels_file": str(test_dir / args.test_labels_filename),
        "include_gold_in_train": args.include_gold_in_train,
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "final_test_accuracy": history[-1]["test_accuracy"],
    }
    save_outputs(
        run_dir=run_dir,
        model=model,
        history=history,
        metrics=metrics,
        per_class_rows=last_per_class_rows,
    )
    print(f"Saved baseline outputs to {run_dir}")


if __name__ == "__main__":
    main()
