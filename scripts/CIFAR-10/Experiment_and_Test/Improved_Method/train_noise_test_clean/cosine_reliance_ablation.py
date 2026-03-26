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

from data_reliance import CifarJsonlRelianceDataset, build_transforms
from utils import set_seed, accuracy, cosine_lr, scale_from_reliance, evaluate
from model import SimpleCNN

def parse_args():
    p = argparse.ArgumentParser("Ablation: cosine vs reliance vs cosine×reliance")

    # Data
    p.add_argument("--train_images", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--test_images", type=str, required=True)
    p.add_argument("--test_jsonl", type=str, required=True)

    p.add_argument("--reliance_jsonl", type=str, required=True)
    p.add_argument("--golden_gold_jsonl", type=str, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)

    # LR
    p.add_argument("--base_lr", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=0.001)

    # Optimizer
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # reliance mapping
    p.add_argument("--map_type", type=str, default="piecewise", choices=["piecewise", "exp"])

    # piecewise
    p.add_argument("--thr_high", type=float, default=0.9)
    p.add_argument("--thr_mid", type=float, default=0.7)
    p.add_argument("--scale_high", type=float, default=1.0)
    p.add_argument("--scale_mid", type=float, default=0.5)
    p.add_argument("--scale_low", type=float, default=0.2)

    # exp
    p.add_argument("--k", type=float, default=4.0)
    p.add_argument("--scale_min", type=float, default=0.2)
    p.add_argument("--scale_max", type=float, default=1.0)

    # Ablation mode
    p.add_argument("--ablation_mode", type=str, required=True, choices=["cosine_only", "reliance_only", "full"])

    p.add_argument("--train_filter_noisy", type=str, default="all", choices=["all", "clean", "noisy"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--save_dir", type=str, default="runs/cosine_reliance_ablation")
    p.add_argument(
        "--metrics_dir",
        type=str,
        default="/users/zeyuhan/charlie_codebase/AI-Critical-Learning/notebooks",
    )

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best.pt"

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Use save_dir name + ablation_mode to avoid overwriting
    run_name = f"{Path(args.save_dir).name}_{args.ablation_mode}"
    metrics_path = metrics_dir / f"{run_name}_metrics.jsonl"
    summary_path = metrics_dir / f"{run_name}_summary.json"
    if metrics_path.exists():
        metrics_path.unlink()

    # Datasets
    train_ds = CifarJsonlRelianceDataset(
        images_root=args.train_images,
        index_jsonl=args.train_jsonl,
        split="train",
        transform=build_transforms(train=True),
        reliance_jsonl=args.reliance_jsonl,
        require_reliance=True,
        exclude_golden_gold_jsonl=args.golden_gold_jsonl,
        filter_noisy=args.train_filter_noisy,
    )
    test_ds = CifarJsonlRelianceDataset(
        images_root=args.test_images,
        index_jsonl=args.test_jsonl,
        split="test",
        transform=build_transforms(train=False),
        reliance_jsonl=None,
        require_reliance=False,
        exclude_golden_gold_jsonl=None,
        filter_noisy=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = SimpleCNN(num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,  # overwritten each step
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    ce = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    print(f"[INFO] ablation_mode={args.ablation_mode}")
    print(f"[INFO] device={device}")
    print(f"[INFO] n_train={len(train_ds)} n_test={len(test_ds)}")
    print(f"[INFO] golden_gold_excluded={bool(args.golden_gold_jsonl)}")
    print(f"[INFO] base_lr={args.base_lr} min_lr={args.min_lr} steps={total_steps}")
    print(f"[INFO] map_type={args.map_type}")
    print(f"[INFO] metrics_jsonl={metrics_path}")

    best_test_acc = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        run_loss = 0.0
        run_acc = 0.0
        run_rbar = 0.0
        run_scale = 0.0
        n_seen = 0

        for x, y, r in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            r_bar = float(r.mean().item())
            scale = scale_from_reliance(r_bar, args)

            lr_t = compute_lr(
                mode=args.ablation_mode,
                step=global_step,
                total_steps=total_steps,
                base_lr=args.base_lr,
                min_lr=args.min_lr,
                scale=scale,
            )
            optimizer.param_groups[0]["lr"] = lr_t

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            run_loss += loss.item() * bs
            run_acc += accuracy(logits, y) * bs
            run_rbar += r_bar * bs
            run_scale += scale * bs
            n_seen += bs
            global_step += 1

        train_loss = run_loss / n_seen
        train_acc = run_acc / n_seen
        train_rbar = run_rbar / n_seen
        train_scale = run_scale / n_seen

        test_loss, test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0
        lr_last = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr(last)={lr_last:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"rbar={train_rbar:.4f} scale={train_scale:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | time={dt:.1f}s"
        )

        row = {
            "epoch": epoch,
            "ablation_mode": args.ablation_mode,
            "lr_last": lr_last,
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "scheduler": args.ablation_mode,
            "map_type": args.map_type,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_rbar": train_rbar,
            "train_scale": train_scale,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "time_sec": dt,
            "n_train": len(train_ds),
            "n_test": len(test_ds),
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "golden_gold_excluded": bool(args.golden_gold_jsonl),
            "golden_gold_jsonl": args.golden_gold_jsonl,
            "reliance_jsonl": args.reliance_jsonl,
            "train_filter_noisy": args.train_filter_noisy,
            "seed": args.seed,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

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
                best_ckpt,
            )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_test_acc": best_test_acc,
                "best_epoch": best_epoch,
                "best_ckpt": str(best_ckpt),
                "metrics_jsonl": str(metrics_path),
                "ablation_mode": args.ablation_mode,
                "n_train": len(train_ds),
                "n_test": len(test_ds),
                "reliance_jsonl": args.reliance_jsonl,
                "golden_gold_jsonl": args.golden_gold_jsonl,
                "map_type": args.map_type,
                "train_filter_noisy": args.train_filter_noisy,
            },
            f,
            indent=2,
        )

    print(f"[DONE] best_test_acc={best_test_acc:.4f} (epoch={best_epoch})")
    print(f"[DONE] metrics saved: {metrics_path}")
    print(f"[DONE] summary saved: {summary_path}")


if __name__ == "__main__":
    main()