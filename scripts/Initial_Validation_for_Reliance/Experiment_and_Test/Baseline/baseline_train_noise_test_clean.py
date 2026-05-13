import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import CifarJsonlDataset, build_transforms
from models import SimpleCNN
from utils import set_seed, accuracy


def parse_args():
    p = argparse.ArgumentParser("CIFAR-10 baseline (SGD + CE) with golden_gold exclusion + metrics saving")

    # Data paths (your provided ones)
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
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    p.add_argument("--min_lr", type=float, default=1e-3)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    # Save checkpoints (local run dir)
    p.add_argument("--save_dir", type=str, default="runs/baseline_sgd")

    # Save metrics (your requested notebooks folder)
    p.add_argument(
        "--metrics_dir",
        type=str,
        default="/users/zeyuhan/charlie_codebase/AI-Critical-Learning/notebooks",
        help="Where to save metrics jsonl + summary json.",
    )

    # Optional: train only clean/noisy subset (baseline default uses all)
    p.add_argument("--train_filter_noisy", type=str, default="all", choices=["all", "clean", "noisy"])

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

    # checkpoint save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"

    # metrics save (notebooks)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_name = Path(args.save_dir).name
    metrics_path = metrics_dir / f"{run_name}_metrics.jsonl"
    summary_path = metrics_dir / f"{run_name}_summary.json"

    # data
    train_loader, n_train = make_loader(
        args.train_images, args.train_jsonl, "train",
        args.batch_size, args.num_workers, train=True,
        train_filter_noisy=args.train_filter_noisy,
        golden_gold_jsonl=args.golden_gold_jsonl,
    )
    test_loader, n_test = make_loader(
        args.test_images, args.test_jsonl, "test",
        args.batch_size, args.num_workers, train=False,
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
    print(f"[INFO] optimizer=SGD lr={args.lr} mom={args.momentum} wd={args.weight_decay} nesterov=True")
    print(f"[INFO] scheduler={args.scheduler}")
    print(f"[INFO] metrics_jsonl={metrics_path}")
    print(f"[INFO] summary_json={summary_path}")
    print(f"[INFO] best_ckpt={best_path}")

    # clear metrics file if exists (so rerun doesn't append)
    if metrics_path.exists():
        metrics_path.unlink()

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

        # write metrics row
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
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "golden_gold_excluded": bool(args.golden_gold_jsonl),
            "golden_gold_jsonl": args.golden_gold_jsonl,
            "train_filter_noisy": args.train_filter_noisy,
            "seed": args.seed,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # save best ckpt
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "best_test_acc": best_test_acc,
                },
                best_path,
            )

    # write summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_test_acc": best_test_acc,
                "best_epoch": best_epoch,
                "best_ckpt": str(best_path),
                "run_dir": str(save_dir),
                "metrics_jsonl": str(metrics_path),
                "n_train": n_train,
                "n_test": n_test,
                "golden_gold_jsonl": args.golden_gold_jsonl,
            },
            f,
            indent=2,
        )

    print(f"[DONE] best_test_acc={best_test_acc:.4f} (epoch={best_epoch})")
    print(f"[DONE] metrics saved: {metrics_path}")
    print(f"[DONE] summary saved: {summary_path}")


if __name__ == "__main__":
    main()