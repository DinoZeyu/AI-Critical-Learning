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
from torch.utils.data import DataLoader, Dataset, Subset
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
GOLD_DATA_DIR = IMAGE_DATA_DIR / "Gold_Data"
RESULTS_ROOT = REPO_ROOT / "Experiments_Results"
DEFAULT_LABELS_FILENAME = "labels.csv"
WITH_GOLD_LABELS_FILENAME = "labels_before_gold_split.csv"
DEFAULT_IMAGE_SIZES = {
    "STL": 96,
    "Flower_102": 224,
}
IMAGE_DATA_ROOT_NAMES = {
    "Train_Clean_Data",
    "Test_Clean_Data",
    "Train_Noise_Data",
    "Test_Noise_Data",
    "Gold_Data",
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
        "--val-dir",
        type=Path,
        help=(
            "Clean validation dataset directory for selected-epoch reporting. Defaults "
            "to Image_Data/Gold_Data/<dataset>, matching gold-guided runs."
        ),
    )
    parser.add_argument(
        "--val-labels-filename",
        default=DEFAULT_LABELS_FILENAME,
        help="Validation labels filename inside --val-dir.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of --val-dir held out as clean validation for selected-epoch reporting "
            "and early stopping. The split is stratified and seed-controlled."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=("val-accuracy", "val-loss"),
        default="val-accuracy",
        help=(
            "Validation metric used for selected-epoch reporting and early "
            "stopping. The diagnostic peak-test epoch is always tracked separately."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many epochs without validation metric improvement. "
            "Default 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation metric improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help=(
            "Save baseline model.pt and model_peak_test.pt. Disabled by default "
            "because baseline runs are for comparison metrics, not model reuse."
        ),
    )
    parser.add_argument(
        "--include-gold-in-train",
        action="store_true",
        help=(
            "For clean train dirs, train with labels_before_gold_split.csv instead "
            "of labels.csv. Gold_Data is still only used for validation unless "
            "--val-dir is changed."
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
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be in (0, 0.5).")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience cannot be negative.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta cannot be negative.")
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


def image_data_relative_parts(path: Path) -> tuple[str, ...] | None:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(IMAGE_DATA_DIR.resolve()).parts
    except ValueError:
        pass

    try:
        repo_relative_parts = resolved_path.relative_to(REPO_ROOT.resolve()).parts
    except ValueError:
        return None

    if "Image_Data" not in repo_relative_parts:
        for index, part in enumerate(repo_relative_parts):
            if part in IMAGE_DATA_ROOT_NAMES:
                return repo_relative_parts[index:]
        return None

    image_data_index = repo_relative_parts.index("Image_Data")
    return repo_relative_parts[image_data_index + 1 :]


def resolve_train_dir(dataset: str, train_dir: Path | None) -> Path:
    if train_dir is not None:
        return resolve_repo_path(train_dir)
    return TRAIN_CLEAN_DIR / dataset


def resolve_test_dir(dataset: str, test_dir: Path | None) -> Path:
    if test_dir is not None:
        return resolve_repo_path(test_dir)
    return TEST_CLEAN_DIR / dataset


def resolve_val_dir(dataset: str, val_dir: Path | None) -> Path:
    if val_dir is not None:
        return resolve_repo_path(val_dir)
    return GOLD_DATA_DIR / dataset


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


def stratified_validation_split(
    rows: list[dict[str, str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    indices_by_label: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        indices_by_label.setdefault(int(row["label"]), []).append(index)

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
    if not val_indices:
        raise ValueError("Validation split is empty; increase --val-fraction or gold data.")
    return train_indices, val_indices


def rows_for_indices(
    rows: list[dict[str, str]],
    indices: list[int],
) -> list[dict[str, str]]:
    return [rows[index] for index in indices]


def make_dataloaders(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    train_labels_filename: str,
    val_labels_filename: str,
    test_labels_filename: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    val_fraction: float,
    seed: int,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    transform = build_transform(image_size)
    train_dataset = CsvImageDataset(
        train_dir,
        labels_filename=train_labels_filename,
        transform=transform,
    )
    val_dataset = CsvImageDataset(
        val_dir,
        labels_filename=val_labels_filename,
        transform=transform,
    )
    test_dataset = CsvImageDataset(
        test_dir,
        labels_filename=test_labels_filename,
        transform=transform,
    )
    val_source_rows = val_dataset.rows
    _, val_indices = stratified_validation_split(
        rows=val_source_rows,
        val_fraction=val_fraction,
        seed=seed,
    )
    val_rows = rows_for_indices(val_source_rows, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
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
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.rows,
        val_rows,
        test_dataset.rows,
    )


def source_group(path: Path, split: str) -> str:
    relative_parts = image_data_relative_parts(path)
    if relative_parts is None:
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
    relative_parts = image_data_relative_parts(path)
    if relative_parts is None:
        relative_parts = path.resolve().relative_to(REPO_ROOT.resolve()).parts

    parts = list(relative_parts)
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
    if experiment_group == "Train_Clean_Test_Clean":
        run_dir = resolve_repo_path(results_root) / experiment_group / dataset
        return run_dir if leaf_path == Path("clean") else run_dir / leaf_path
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
    peak_test_state: dict[str, torch.Tensor] | None,
    peak_test_per_class_rows: list[dict[str, Any]],
    save_checkpoints: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if save_checkpoints:
        torch.save(model.state_dict(), run_dir / "model.pt")
    if save_checkpoints and peak_test_state is not None:
        torch.save(peak_test_state, run_dir / "model_peak_test.pt")
    write_csv(run_dir / "history.csv", history)
    write_csv(run_dir / "per_class_accuracy.csv", per_class_rows)
    write_csv(run_dir / "per_class_accuracy_selected.csv", per_class_rows)
    write_csv(run_dir / "per_class_accuracy_peak_test.csv", peak_test_per_class_rows)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    train_dir = resolve_train_dir(args.dataset, args.train_dir)
    val_dir = resolve_val_dir(args.dataset, args.val_dir)
    test_dir = resolve_test_dir(args.dataset, args.test_dir)
    train_labels_filename = resolve_train_labels_filename(args)
    image_size = resolve_image_size(args.dataset, args.image_size)

    train_loader, val_loader, test_loader, train_rows, val_rows, test_rows = make_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        train_labels_filename=train_labels_filename,
        val_labels_filename=args.val_labels_filename,
        test_labels_filename=args.test_labels_filename,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    num_classes = infer_num_classes(train_rows, val_rows, test_rows)
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
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    test_accuracy_at_best_validation = -1.0
    test_loss_at_best_validation = float("inf")
    best_state = None
    best_epoch = 0
    last_per_class_rows = []
    best_per_class_rows = []
    epochs_without_improvement = 0
    stopped_epoch = 0
    max_test_accuracy = -1.0
    max_test_loss_at_peak = float("inf")
    max_test_epoch = 0
    peak_test_state = None
    peak_test_per_class_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        val_loss, val_accuracy, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
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
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        history.append(row)
        last_per_class_rows = per_class_rows
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}"
        )

        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            max_test_loss_at_peak = test_loss
            max_test_epoch = epoch
            peak_test_per_class_rows = per_class_rows
            peak_test_state = clone_state_dict(model)

        if args.selection_metric == "val-accuracy":
            metric_improved = (
                val_accuracy > best_validation_accuracy + args.early_stop_min_delta
            )
        else:
            metric_improved = val_loss < best_validation_loss - args.early_stop_min_delta

        if metric_improved:
            best_validation_accuracy = val_accuracy
            best_validation_loss = val_loss
            test_accuracy_at_best_validation = test_accuracy
            test_loss_at_best_validation = test_loss
            best_epoch = epoch
            best_per_class_rows = per_class_rows
            epochs_without_improvement = 0
            best_state = clone_state_dict(model)
        else:
            epochs_without_improvement += 1

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            stopped_epoch = epoch
            print(
                f"Early stopping baseline at epoch={epoch}; "
                f"selection_metric={args.selection_metric} "
                f"selected_epoch={best_epoch} "
                f"selected_val_loss={best_validation_loss:.4f} "
                f"best_val_acc={best_validation_accuracy:.4f} "
                f"test_acc_at_selected={test_accuracy_at_best_validation:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "test_dir": str(test_dir),
        "train_labels_file": str(train_dir / train_labels_filename),
        "val_labels_file": str(val_dir / args.val_labels_filename),
        "test_labels_file": str(test_dir / args.test_labels_filename),
        "include_gold_in_train": args.include_gold_in_train,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "validation_source": "gold_data_stratified_split",
        "validation_fraction": args.val_fraction,
        "test_examples": len(test_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "selection_metric": args.selection_metric,
        "selected_epoch_policy": f"best_clean_validation_{args.selection_metric}",
        "selected_checkpoint_policy": (
            f"best_clean_validation_{args.selection_metric}"
            if args.save_checkpoints
            else None
        ),
        "checkpoints_saved": args.save_checkpoints,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stopped": stopped_epoch > 0,
        "stopped_epoch": stopped_epoch,
        "seed": args.seed,
        "device": str(device),
        "selected_epoch": best_epoch,
        "selected_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "test_loss_at_selected_epoch": test_loss_at_best_validation,
        "test_accuracy_at_selected_epoch": test_accuracy_at_best_validation,
        "peak_test_checkpoint_policy": "diagnostic_oracle_best_clean_test_accuracy",
        "max_test_accuracy_observed": max_test_accuracy,
        "max_test_loss_at_peak": max_test_loss_at_peak,
        "max_test_epoch_observed": max_test_epoch,
        "selected_checkpoint": str(run_dir / "model.pt") if args.save_checkpoints else None,
        "peak_test_checkpoint": (
            str(run_dir / "model_peak_test.pt") if args.save_checkpoints else None
        ),
        "best_epoch": max_test_epoch,
        "best_test_accuracy": max_test_accuracy,
        "best_test_epoch": max_test_epoch,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "final_test_loss": history[-1]["test_loss"],
    }
    save_outputs(
        run_dir=run_dir,
        model=model,
        history=history,
        metrics=metrics,
        per_class_rows=best_per_class_rows or last_per_class_rows,
        peak_test_state=peak_test_state,
        peak_test_per_class_rows=peak_test_per_class_rows,
        save_checkpoints=args.save_checkpoints,
    )
    print(f"Saved baseline outputs to {run_dir}")


if __name__ == "__main__":
    main()
