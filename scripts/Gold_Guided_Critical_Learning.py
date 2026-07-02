from __future__ import annotations

import argparse
import csv
import json
import math
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

from cl.methods.gold_guided_critical import (  # noqa: E402
    compute_gold_prototypes,
    cycle_loader,
    freeze_model,
    train_critical_epoch,
)
from cl.model import SimpleCNN  # noqa: E402


IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"
GOLD_DATA_DIR = IMAGE_DATA_DIR / "Gold_Data"
TEST_CLEAN_DIR = IMAGE_DATA_DIR / "Test_Clean_Data"
RESULTS_ROOT = REPO_ROOT / "Experiments_Results"
GOLD_EVALUATOR_ROOT = REPO_ROOT / "Gold_Evaluators"
DEFAULT_LABELS_FILENAME = "labels.csv"
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
            "Gold-guided critical learning using a pre-trained, frozen gold "
            "evaluator checkpoint and a separate mixed-data learner."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. STL.")
    parser.add_argument(
        "--mixed-train-dir",
        type=Path,
        required=True,
        help="Mixed/noisy training dataset directory with images/ and labels.csv.",
    )
    parser.add_argument(
        "--gold-evaluator-checkpoint",
        type=Path,
        help=(
            "Frozen gold evaluator checkpoint. Defaults to "
            "Gold_Evaluators/<dataset>/model.pt."
        ),
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        help=(
            "Gold dataset directory used for prototypes and stability batches. "
            "Defaults to Image_Data/Gold_Data/<dataset>."
        ),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Clean test dataset directory. Defaults to Image_Data/Test_Clean_Data/<dataset>.",
    )
    parser.add_argument("--mixed-labels-filename", default=DEFAULT_LABELS_FILENAME)
    parser.add_argument("--gold-labels-filename", default=DEFAULT_LABELS_FILENAME)
    parser.add_argument("--test-labels-filename", default=DEFAULT_LABELS_FILENAME)
    parser.add_argument(
        "--run-name",
        help=(
            "Optional output leaf folder. Defaults to the mixed train directory "
            "path under Image_Data."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--gold-batch-size",
        type=int,
        help="Gold batch size for prototypes and stability loss. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of Gold_Data held out as clean validation for model "
            "selection and early stopping."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--base-lr",
        type=float,
        help=(
            "Initial learner LR for cosine decay. If set, it overrides --lr "
            "as the optimizer starting LR."
        ),
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-3,
        help="Minimum learner LR for cosine decay when --base-lr is set.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd"),
        default="adamw",
        help="Optimizer for the learner.",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument(
        "--alpha-start",
        type=float,
        default=0.5,
        help=(
            "Starting controller sharpness for dynamic alpha schedules. "
            "--alpha remains the final sharpness."
        ),
    )
    parser.add_argument(
        "--alpha-schedule",
        choices=("fixed", "linear", "cosine"),
        default="fixed",
        help=(
            "Schedule for alpha_k in the non-negative controller derived from "
            "tanh(alpha_k*(r_i-tau)). fixed uses --alpha every epoch; "
            "linear/cosine increase from --alpha-start to --alpha over the "
            "training run."
        ),
    )
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument(
        "--controller-mode",
        choices=("positive", "clamped"),
        default="positive",
        help=(
            "How to map the signed tanh controller into a non-negative sample "
            "weight. positive maps tanh to [0, 1]; clamped keeps only the "
            "non-negative part of tanh. Negative anti-fitting is intentionally "
            "not used."
        ),
    )
    parser.add_argument(
        "--min-control",
        type=float,
        default=0.0,
        help=(
            "Lower bound for controller output. Default 0 means low-consistency "
            "samples can be skipped, but never negatively fitted."
        ),
    )
    parser.add_argument("--lambda-gold", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
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
        "--selection-metric",
        choices=("val-accuracy", "val-loss"),
        default="val-accuracy",
        help=(
            "Validation metric used for model.pt checkpoint selection and early "
            "stopping. The diagnostic peak-test checkpoint is always tracked separately."
        ),
    )
    parser.add_argument(
        "--save-gold-artifacts",
        action="store_true",
        help=(
            "Also copy the frozen gold evaluator and computed prototypes into "
            "the run directory. Disabled by default because these are shared "
            "across runs with the same dataset, gold split, and evaluator checkpoint."
        ),
    )
    parser.add_argument(
        "--freeze-learner-batch-norm",
        action="store_true",
        help=(
            "Keep learner BatchNorm layers in eval mode during mixed-data "
            "training so clean/gold-domain running statistics are not updated "
            "by noisy feature-domain batches. Affine BN parameters remain trainable."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-size",
        type=int,
        help="Square input size. Defaults to 96 for STL and 224 for Flower_102.",
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
    if args.gold_batch_size is not None and args.gold_batch_size <= 0:
        raise ValueError("--gold-batch-size must be positive.")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be in (0, 0.5).")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.base_lr is not None and args.base_lr <= 0:
        raise ValueError("--base-lr must be positive.")
    if args.min_lr <= 0:
        raise ValueError("--min-lr must be positive.")
    if args.base_lr is not None and args.min_lr > args.base_lr:
        raise ValueError("--min-lr cannot be greater than --base-lr.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if not 0.0 <= args.momentum < 1.0:
        raise ValueError("--momentum must be in [0, 1).")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must be in [0, 1].")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive.")
    if args.alpha_start <= 0:
        raise ValueError("--alpha-start must be positive.")
    if not 0.0 <= args.tau <= 1.0:
        raise ValueError("--tau must be in [0, 1].")
    if args.lambda_gold < 0:
        raise ValueError("--lambda-gold cannot be negative.")
    if args.grad_clip is not None and args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive.")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience cannot be negative.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta cannot be negative.")
    if not 0.0 <= args.min_control <= 1.0:
        raise ValueError("--min-control must be in [0, 1].")
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


def resolve_test_dir(dataset: str, test_dir: Path | None) -> Path:
    if test_dir is not None:
        return resolve_repo_path(test_dir)
    return TEST_CLEAN_DIR / dataset


def resolve_gold_evaluator_checkpoint(
    dataset: str,
    checkpoint: Path | None,
) -> Path:
    if checkpoint is not None:
        return resolve_repo_path(checkpoint)
    return GOLD_EVALUATOR_ROOT / dataset / "model.pt"


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
    all_indices = list(range(len(rows)))
    if val_fraction == 0:
        return all_indices, []

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


def alpha_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if args.alpha_schedule == "fixed":
        return args.alpha

    if args.epochs == 1:
        progress = 1.0
    else:
        progress = (epoch - 1) / (args.epochs - 1)

    if args.alpha_schedule == "linear":
        scale = progress
    elif args.alpha_schedule == "cosine":
        scale = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown alpha schedule: {args.alpha_schedule}")

    return args.alpha_start + (args.alpha - args.alpha_start) * scale


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


def build_run_dir(
    results_root: Path,
    dataset: str,
    mixed_train_dir: Path,
    test_dir: Path,
    run_name: str | None,
) -> Path:
    experiment_group = f"{source_group(mixed_train_dir, 'Train')}_{source_group(test_dir, 'Test')}"
    leaf_path = Path(run_name) if run_name is not None else relative_result_path(mixed_train_dir)
    return (
        resolve_repo_path(results_root)
        / experiment_group
        / "Gold_Guided_Critical_Learning"
        / dataset
        / leaf_path
    )


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


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing gold evaluator checkpoint: {checkpoint_path}. "
            "Run scripts/Train_Gold_Evaluator.py first."
        )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def save_outputs(
    run_dir: Path,
    learner: nn.Module,
    evaluator: nn.Module,
    prototypes: torch.Tensor,
    class_counts: torch.Tensor,
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    per_class_rows: list[dict[str, Any]],
    peak_test_state: dict[str, torch.Tensor] | None,
    peak_test_per_class_rows: list[dict[str, Any]],
    save_gold_artifacts: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(learner.state_dict(), run_dir / "model.pt")
    if peak_test_state is not None:
        torch.save(peak_test_state, run_dir / "model_peak_test.pt")
    if save_gold_artifacts:
        torch.save(evaluator.state_dict(), run_dir / "frozen_gold_evaluator.pt")
        torch.save(
            {
                "prototypes": prototypes.detach().cpu(),
                "class_counts": class_counts.detach().cpu(),
            },
            run_dir / "gold_prototypes.pt",
        )
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

    mixed_train_dir = resolve_repo_path(args.mixed_train_dir)
    gold_dir = resolve_gold_dir(args.dataset, args.gold_dir)
    test_dir = resolve_test_dir(args.dataset, args.test_dir)
    gold_evaluator_checkpoint = resolve_gold_evaluator_checkpoint(
        dataset=args.dataset,
        checkpoint=args.gold_evaluator_checkpoint,
    )
    image_size = resolve_image_size(args.dataset, args.image_size)
    gold_batch_size = args.gold_batch_size or args.batch_size
    device = torch.device(args.device)
    transform = build_transform(image_size)

    gold_dataset = CsvImageDataset(
        dataset_dir=gold_dir,
        labels_filename=args.gold_labels_filename,
        transform=transform,
    )
    gold_rows = gold_dataset.rows
    gold_train_indices, gold_val_indices = stratified_gold_split(
        rows=gold_rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    gold_train_dataset = Subset(gold_dataset, gold_train_indices)
    gold_val_dataset = Subset(gold_dataset, gold_val_indices)
    gold_train_rows = rows_for_indices(gold_rows, gold_train_indices)
    gold_val_rows = rows_for_indices(gold_rows, gold_val_indices)

    gold_loader = make_dataloader(
        dataset=gold_train_dataset,
        batch_size=gold_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    gold_eval_loader = make_dataloader(
        dataset=gold_train_dataset,
        batch_size=gold_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_loader = make_dataloader(
        dataset=gold_val_dataset,
        batch_size=gold_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    mixed_loader, mixed_rows = make_loader(
        dataset_dir=mixed_train_dir,
        labels_filename=args.mixed_labels_filename,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    test_loader, test_rows = make_loader(
        dataset_dir=test_dir,
        labels_filename=args.test_labels_filename,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    num_classes = infer_num_classes(gold_rows, mixed_rows, test_rows)
    criterion = nn.CrossEntropyLoss()

    evaluator = SimpleCNN(num_classes=num_classes).to(device)
    load_checkpoint(
        model=evaluator,
        checkpoint_path=gold_evaluator_checkpoint,
        device=device,
    )
    freeze_model(evaluator)

    learner = SimpleCNN(num_classes=num_classes).to(device)
    learner.load_state_dict(evaluator.state_dict())

    prototypes, class_counts = compute_gold_prototypes(
        evaluator=evaluator,
        gold_loader=gold_eval_loader,
        num_classes=num_classes,
        device=device,
    )
    gold_batch_iterator = cycle_loader(gold_loader)
    optimizer_lr = args.base_lr if args.base_lr is not None else args.lr
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            learner.parameters(),
            lr=optimizer_lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            learner.parameters(),
            lr=optimizer_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    scheduler = None
    if args.base_lr is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    run_dir = build_run_dir(
        results_root=args.results_root,
        dataset=args.dataset,
        mixed_train_dir=mixed_train_dir,
        test_dir=test_dir,
        run_name=args.run_name,
    )
    history = []
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    test_accuracy_at_best_validation = -1.0
    test_loss_at_best_validation = float("inf")
    best_epoch = 0
    best_state = None
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
        current_lr = optimizer.param_groups[0]["lr"]
        current_alpha = alpha_for_epoch(args, epoch)
        train_metrics = train_critical_epoch(
            learner=learner,
            evaluator=evaluator,
            mixed_loader=mixed_loader,
            gold_batch_iterator=gold_batch_iterator,
            prototypes=prototypes,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            beta=args.beta,
            alpha=current_alpha,
            tau=args.tau,
            lambda_gold=args.lambda_gold,
            controller_mode=args.controller_mode,
            min_control=args.min_control,
            grad_clip=args.grad_clip,
            freeze_learner_batch_norm=args.freeze_learner_batch_norm,
        )
        val_loss, val_accuracy, _ = evaluate(
            model=learner,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
        test_loss, test_accuracy, per_class_rows = evaluate(
            model=learner,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "alpha": current_alpha,
            **train_metrics,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        history.append(row)
        last_per_class_rows = per_class_rows
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"mixed_loss={train_metrics['mixed_loss']:.4f} "
            f"gold_stability_loss={train_metrics['gold_stability_loss']:.4f} "
            f"train_acc={train_metrics['train_accuracy']:.4f} "
            f"mean_gold_consistency={train_metrics['mean_gold_consistency']:.4f} "
            f"mean_control={train_metrics['mean_control']:.4f} "
            f"alpha={current_alpha:.4f} "
            f"lr={current_lr:.6f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}"
        )

        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            max_test_loss_at_peak = test_loss
            max_test_epoch = epoch
            peak_test_per_class_rows = per_class_rows
            peak_test_state = clone_state_dict(learner)

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
            best_state = clone_state_dict(learner)
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            stopped_epoch = epoch
            print(
                f"Early stopping at epoch={epoch}; "
                f"selection_metric={args.selection_metric} "
                f"best_epoch={best_epoch} "
                f"best_val_loss={best_validation_loss:.4f} "
                f"best_val_acc={best_validation_accuracy:.4f} "
                f"test_acc_at_best_val={test_accuracy_at_best_validation:.4f}"
            )
            break

    if best_state is not None:
        learner.load_state_dict(best_state)

    metrics = {
        "method": "gold_guided_critical_learning",
        "dataset": args.dataset,
        "num_classes": num_classes,
        "gold_evaluator_checkpoint": str(gold_evaluator_checkpoint),
        "gold_dir": str(gold_dir),
        "mixed_train_dir": str(mixed_train_dir),
        "test_dir": str(test_dir),
        "gold_labels_file": str(gold_dir / args.gold_labels_filename),
        "mixed_labels_file": str(mixed_train_dir / args.mixed_labels_filename),
        "test_labels_file": str(test_dir / args.test_labels_filename),
        "gold_examples": len(gold_rows),
        "gold_train_examples": len(gold_train_rows),
        "gold_validation_examples": len(gold_val_rows),
        "validation_source": "gold_data_stratified_split",
        "validation_fraction": args.val_fraction,
        "mixed_train_examples": len(mixed_rows),
        "test_examples": len(test_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gold_batch_size": gold_batch_size,
        "image_size": image_size,
        "lr": args.lr,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "optimizer_start_lr": optimizer_lr,
        "lr_scheduler": "cosine" if scheduler is not None else None,
        "weight_decay": args.weight_decay,
        "beta": args.beta,
        "alpha": args.alpha,
        "alpha_start": args.alpha_start,
        "alpha_schedule": args.alpha_schedule,
        "tau": args.tau,
        "controller_mode": args.controller_mode,
        "min_control": args.min_control,
        "lambda_gold": args.lambda_gold,
        "grad_clip": args.grad_clip,
        "save_gold_artifacts": args.save_gold_artifacts,
        "freeze_learner_batch_norm": args.freeze_learner_batch_norm,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "selection_metric": args.selection_metric,
        "selected_checkpoint_policy": f"best_gold_validation_{args.selection_metric}",
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
        "peak_test_checkpoint": str(run_dir / "model_peak_test.pt"),
        "final_test_accuracy": history[-1]["test_accuracy"],
        "final_test_loss": history[-1]["test_loss"],
    }
    save_outputs(
        run_dir=run_dir,
        learner=learner,
        evaluator=evaluator,
        prototypes=prototypes,
        class_counts=class_counts,
        history=history,
        metrics=metrics,
        per_class_rows=best_per_class_rows or last_per_class_rows,
        peak_test_state=peak_test_state,
        peak_test_per_class_rows=peak_test_per_class_rows,
        save_gold_artifacts=args.save_gold_artifacts,
    )
    print(f"Saved gold-guided critical learning outputs to {run_dir}")


if __name__ == "__main__":
    main()
