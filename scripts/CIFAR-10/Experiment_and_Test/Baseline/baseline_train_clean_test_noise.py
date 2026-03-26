import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import CifarJsonlDataset, build_transforms
from models import SimpleCNN
from utils import set_seed, accuracy


def parse_args():
    p = argparse.ArgumentParser("CIFAR-10 baseline (SGD + cosine) for train clean / test noise")

    # Data paths
    p.add_argument("--train_images", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--test_images", type=str, required=True)
    p.add_argument("--test_jsonl", type=str, required=True)

    # Golden gold jsonl (exclude from TRAIN only)
    p.add_argument(
        "--golden_gold_jsonl",
        type=str,
        default=None,
        help="Path to Golden_Gold_Standard_seed42.jsonl. If provided, its raw_id will be excluded from TRAIN.",
    )

    # Training config
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)

    # Optimizer config
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # Scheduler config
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--min_lr", type=float, default=1e-3)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    # Metrics output
    p.add_argument(
        "--metrics_dir",
        type=str,
        default="/users/zeyuhan/charlie_codebase/AI-Critical-Learning/notebooks/Improved_Method/train_clean_test_noise",
    )
    p.add_argument("--metrics_name", type=str, default="baseline_sgd_metrics.jsonl")

    # Optional: train only clean/noisy subset
    p.add_argument("--train_filter_noisy", type=str, default="clean", choices=["all", "clean", "noisy"])

    return p.parse_args()


def make_loader(
    images_root: str,
    jsonl: str,
    split: str,
    batch_size: int,
    num_workers: int,
    train: bool,
    train_filter_noisy: str = "all",
    golden_gold_jsonl: str | None = None,
):
    tfm = build_transforms(train=train)

    if train_filter_noisy == "all":
        filter_noisy = None
    elif train_filter_noisy == "clean":
        filter_noisy = False
    else:
        filter_noisy = True

    ds = CifarJsonlDataset(
        images_root=images_root,
        index_jsonl=jsonl,
        split=split,
        transform=tfm,
        filter_noisy=filter_noisy if train else None,
        exclude_golden_gold_jsonl=golden_gold_jsonl if train else None,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
    return loader, len(ds)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)

        total_loss += ce_sum(logits, y).item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += y.numel()

    return total_loss / total_n, total_correct / total_n


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device):
    model.train()
    ce = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, y) * bs
        n += bs

    return running_loss / n, running_acc / n


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / args.metrics_name

    if metrics_path.exists():
        metrics_path.unlink()

    train_loader, n_train = make_loader(
        args.train_images,
        args.train_jsonl,
        "train",
        args.batch_size,
        args.num_workers,
        train=True,
        train_filter_noisy=args.train_filter_noisy,
        golden_gold_jsonl=args.golden_gold_jsonl,
    )
    test_loader, n_test = make_loader(
        args.test_images,
        args.test_jsonl,
        "test",
        args.batch_size,
        args.num_workers,
        train=False,
        train_filter_noisy="all",
        golden_gold_jsonl=None,
    )

    model = SimpleCNN(num_classes=10).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    else:
        scheduler = None

    print(f"[INFO] device={device}")
    print(f"[INFO] train={n_train} test={n_test}")
    if args.golden_gold_jsonl:
        print(f"[INFO] Excluding golden_gold from TRAIN using: {args.golden_gold_jsonl}")
    print(f"[INFO] train_filter_noisy={args.train_filter_noisy}")
    print(f"[INFO] optimizer=SGD lr={args.lr} mom={args.momentum} wd={args.weight_decay} nesterov=True")
    print(f"[INFO] scheduler={args.scheduler}")
    print(f"[INFO] test_images={args.test_images}")
    print(f"[INFO] test_jsonl={args.test_jsonl}")
    print(f"[INFO] metrics_jsonl={metrics_path}")

    best_test_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={lr_now:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"time={dt:.1f}s"
        )

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "time_sec": dt,
            "n_train": n_train,
            "n_test": n_test,
            "scheduler": args.scheduler,
            "base_lr": args.lr,
            "min_lr": args.min_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "golden_gold_excluded": bool(args.golden_gold_jsonl),
            "golden_gold_jsonl": args.golden_gold_jsonl,
            "train_filter_noisy": args.train_filter_noisy,
            "train_images": args.train_images,
            "train_jsonl": args.train_jsonl,
            "test_images": args.test_images,
            "test_jsonl": args.test_jsonl,
            "seed": args.seed,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

    print(f"[DONE] best_test_acc={best_test_acc:.4f} (epoch={best_epoch})")
    print(f"[DONE] metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()