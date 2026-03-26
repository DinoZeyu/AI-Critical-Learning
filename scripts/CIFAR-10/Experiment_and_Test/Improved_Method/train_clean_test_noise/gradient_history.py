import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_reliance import CifarJsonlRelianceDataset, build_transforms
from utils import set_seed, accuracy, cosine_lr, evaluate
from model import SimpleCNN


def parse_args():
    p = argparse.ArgumentParser("v8: reliance + gradient history direction consistency")

    # Data paths
    p.add_argument("--train_images", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--test_images", type=str, required=True)
    p.add_argument("--test_jsonl", type=str, required=True)

    # Reliance file
    p.add_argument("--reliance_jsonl", type=str, required=True)

    # Exclude golden_gold from TRAIN
    p.add_argument("--golden_gold_jsonl", type=str, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)

    # Base cosine params
    p.add_argument("--base_lr", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=0.001)

    # Optimizer
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # Gradient history
    p.add_argument("--grad_mu", type=float, default=0.9, help="EMA factor for gradient history H_t")
    p.add_argument("--alpha_rel", type=float, default=1.0, help="weight for reliance risk term a*(1-r_bar)")
    p.add_argument("--beta_dir", type=float, default=1.0, help="weight for direction divergence term b*D_dir")
    p.add_argument("--dir_gamma", type=float, default=2.0, help="strength in exp(-gamma * D_hat)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    # Metrics output
    p.add_argument(
        "--metrics_dir",
        type=str,
        default="/users/zeyuhan/charlie_codebase/AI-Critical-Learning/notebooks/Improved_Method/train_noise_test_noise",
    )
    p.add_argument("--metrics_name", type=str, default="gradient_history_metrics.jsonl")

    # Optional: filter noisy in train ("all"|"clean"|"noisy")
    p.add_argument("--train_filter_noisy", type=str, default="all", choices=["all", "clean", "noisy"])

    args = p.parse_args()

    if not (0.0 <= args.grad_mu < 1.0):
        raise ValueError("--grad_mu must satisfy 0 <= grad_mu < 1")
    if args.alpha_rel < 0.0:
        raise ValueError("--alpha_rel must be >= 0")
    if args.beta_dir < 0.0:
        raise ValueError("--beta_dir must be >= 0")
    if args.dir_gamma < 0.0:
        raise ValueError("--dir_gamma must be >= 0")

    return args


def flatten_grads(model: nn.Module) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if p.grad is None:
            continue
        chunks.append(p.grad.detach().reshape(-1))
    if not chunks:
        raise RuntimeError("No gradients found when flattening gradients.")
    return torch.cat(chunks, dim=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / args.metrics_name

    if metrics_path.exists():
        metrics_path.unlink()

    # datasets
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
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    ce = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    print(f"[INFO] device={device}")
    print(f"[INFO] n_train={len(train_ds)} n_test={len(test_ds)}")
    print(f"[INFO] golden_gold_excluded={bool(args.golden_gold_jsonl)}")
    print(f"[INFO] train_filter_noisy={args.train_filter_noisy}")
    print(f"[INFO] base_lr={args.base_lr} min_lr={args.min_lr} cosine_steps={total_steps}")
    print(
        f"[INFO] grad_mu={args.grad_mu} alpha_rel={args.alpha_rel} "
        f"beta_dir={args.beta_dir} dir_gamma={args.dir_gamma}"
    )
    print(f"[INFO] test_images={args.test_images}")
    print(f"[INFO] test_jsonl={args.test_jsonl}")
    print(f"[INFO] metrics_jsonl={metrics_path}")

    best_test_acc = -1.0
    best_epoch = -1
    global_step = 0

    # Gradient history H_t
    grad_hist = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        run_loss = 0.0
        run_acc = 0.0
        run_rbar = 0.0
        run_base_lr = 0.0
        run_cos_sim = 0.0
        run_d_dir = 0.0
        run_d_hat = 0.0
        run_scale = 0.0
        run_effective_lr = 0.0
        n_seen = 0

        last_base_lr_t = args.base_lr
        last_cos_sim_t = 1.0
        last_d_dir_t = 0.0
        last_d_hat_t = 0.0
        last_scale_t = 1.0
        last_lr_t = args.base_lr

        for x, y, r in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            r_bar = float(r.mean().item())

            base_lr_t = cosine_lr(global_step, total_steps, args.base_lr, args.min_lr)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()

            g_t = flatten_grads(model)

            if grad_hist is None:
                cos_sim_t = 1.0
                d_dir_t = 0.0
                grad_hist = torch.zeros_like(g_t)
            else:
                g_norm = torch.norm(g_t, p=2)
                h_norm = torch.norm(grad_hist, p=2)

                if g_norm.item() == 0.0 or h_norm.item() == 0.0:
                    cos_sim_t = 1.0
                    d_dir_t = 0.0
                else:
                    cos_sim_t = float(F.cosine_similarity(g_t, grad_hist, dim=0).item())
                    cos_sim_t = max(-1.0, min(1.0, cos_sim_t))
                    d_dir_t = 0.5 * (1.0 - cos_sim_t)

            d_hat_t = args.alpha_rel * (1.0 - r_bar) + args.beta_dir * d_dir_t
            scale_t = math.exp(-args.dir_gamma * d_hat_t)
            lr_t = base_lr_t * scale_t
            optimizer.param_groups[0]["lr"] = lr_t

            optimizer.step()

            # update gradient history after using current g_t
            grad_hist = args.grad_mu * grad_hist + (1.0 - args.grad_mu) * g_t.detach()

            bs = y.size(0)
            run_loss += loss.item() * bs
            run_acc += accuracy(logits, y) * bs
            run_rbar += r_bar * bs
            run_base_lr += base_lr_t * bs
            run_cos_sim += cos_sim_t * bs
            run_d_dir += d_dir_t * bs
            run_d_hat += d_hat_t * bs
            run_scale += scale_t * bs
            run_effective_lr += lr_t * bs
            n_seen += bs
            global_step += 1

            last_base_lr_t = base_lr_t
            last_cos_sim_t = cos_sim_t
            last_d_dir_t = d_dir_t
            last_d_hat_t = d_hat_t
            last_scale_t = scale_t
            last_lr_t = lr_t

        train_loss = run_loss / n_seen
        train_acc = run_acc / n_seen
        train_rbar = run_rbar / n_seen
        train_base_lr = run_base_lr / n_seen
        train_cos_sim = run_cos_sim / n_seen
        train_d_dir = run_d_dir / n_seen
        train_d_hat = run_d_hat / n_seen
        train_scale = run_scale / n_seen
        train_effective_lr = run_effective_lr / n_seen

        test_loss, test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr(last)={last_lr_t:.6f} | "
            f"cos(last)={last_cos_sim_t:.4f} d_dir(last)={last_d_dir_t:.4f} d_hat(last)={last_d_hat_t:.4f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} rbar={train_rbar:.4f} d_dir={train_d_dir:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | time={dt:.1f}s"
        )

        row = {
            "epoch": epoch,
            "lr_last": last_lr_t,
            "base_lr_last": last_base_lr_t,
            "cos_sim_last": last_cos_sim_t,
            "d_dir_last": last_d_dir_t,
            "d_hat_last": last_d_hat_t,
            "scale_last": last_scale_t,
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "scheduler": "gradient_histroy",
            "grad_mu": args.grad_mu,
            "alpha_rel": args.alpha_rel,
            "beta_dir": args.beta_dir,
            "dir_gamma": args.dir_gamma,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_rbar": train_rbar,
            "train_base_lr": train_base_lr,
            "train_cos_sim": train_cos_sim,
            "train_d_dir": train_d_dir,
            "train_d_hat": train_d_hat,
            "train_scale": train_scale,
            "train_effective_lr": train_effective_lr,
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
            "test_images": args.test_images,
            "test_jsonl": args.test_jsonl,
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