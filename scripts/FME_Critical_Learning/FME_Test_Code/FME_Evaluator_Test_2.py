import argparse
import copy
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()

CANDIDATE_IMPORT_DIRS = [
    THIS_FILE.parent,
    THIS_FILE.parent.parent,
    Path("/users/zeyuhan/charlie_codebase/AI-Critical-Learning/scripts/FME_Critical_Learning"),
    Path("/users/zeyuhan/charlie_codebase/AI-Critical-Learning/scripts/CIFAR-10/Experiment_and_Test/Improved_Method/Function_4"),
]

for p in CANDIDATE_IMPORT_DIRS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------
# Shared project imports
# ---------------------------------------------------------------------
from data_reliance import CifarJsonlRelianceDataset, build_transforms
from utils import set_seed, accuracy, cosine_lr, evaluate
from model import SimpleCNN


# ---------------------------------------------------------------------
# Reuse Test_1 components
# ---------------------------------------------------------------------
from FME_Evaluator_Test_1 import (
    GoldJsonlDataset,
    FeatureHook,
    compute_gold_prototypes,
    compute_gold_consistency,
    controller_from_reliability,
    stable_controlled_losses,
    train_gold_reference,
)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        "FME Evaluator Test 2: Gold-Guided Critical Learning + Data Reliance"
    )

    # Data paths
    p.add_argument("--train_images", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--test_images", type=str, required=True)
    p.add_argument("--test_jsonl", type=str, required=True)

    # This is required in Test_2.
    # It provides data_reliance from reliance_raw10k_seed42.jsonl.
    p.add_argument("--reliance_jsonl", type=str, required=True)

    # Gold data
    p.add_argument("--golden_gold_jsonl", type=str, required=True)

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gold_batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)

    # Gold reference training
    p.add_argument("--gold_epochs", type=int, default=10)
    p.add_argument("--gold_lr", type=float, default=0.05)
    p.add_argument("--gold_min_lr", type=float, default=0.001)
    p.add_argument("--gold_scheduler", type=str, default="cosine", choices=["constant", "cosine"])
    p.add_argument("--eval_during_gold", action="store_true")

    # Mixed training LR
    p.add_argument("--base_lr", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=0.001)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["constant", "cosine"])

    # Optimizer
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # Gold consistency:
    # r_gold = beta * r_prob + (1 - beta) * r_sim
    p.add_argument("--beta", type=float, default=0.2)

    # Hybrid reliability:
    # r = gamma_gold * r_gold + (1 - gamma_gold) * r_llm
    # Here r_llm is data_reliance.
    p.add_argument("--gamma_gold", type=float, default=0.5)

    # Controller
    p.add_argument("--tau", type=float, default=0.45)
    p.add_argument("--alpha", type=float, default=2.0)

    # Gold stability
    p.add_argument("--lambda_g", type=float, default=0.1)

    # Anti-fitting.
    # For the current best stable version, keep this as 0.0.
    p.add_argument("--anti_weight", type=float, default=0.0)

    # Optional stability
    p.add_argument("--grad_clip", type=float, default=5.0)

    # Optional noisy filter for mixed train data
    p.add_argument("--train_filter_noisy", type=str, default="all", choices=["all", "clean", "noisy"])

    # Runtime
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    # Save
    p.add_argument(
        "--result_dir",
        type=str,
        default="/users/zeyuhan/charlie_codebase/AI-Critical-Learning/scripts/FME_Critical_Learning/Result",
    )
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--metrics_dir", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.save_dir is None:
        save_dir = result_dir / "FME_Test_2"
    else:
        save_dir = Path(args.save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_dir is None:
        metrics_dir = result_dir
    else:
        metrics_dir = Path(args.metrics_dir)

    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_name = save_dir.name

    best_ckpt = save_dir / "best.pt"
    final_ckpt = save_dir / "final.pt"
    gold_ckpt = save_dir / "gold_reference.pt"

    metrics_path = metrics_dir / f"{run_name}_metrics.jsonl"
    summary_path = metrics_dir / f"{run_name}_summary.json"
    config_path = metrics_dir / f"{run_name}_config.json"

    if metrics_path.exists():
        metrics_path.unlink()

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # -------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------
    gold_train_ds = GoldJsonlDataset(
        images_root=args.train_images,
        gold_jsonl=args.golden_gold_jsonl,
        transform=build_transforms(train=True),
    )

    gold_eval_ds = GoldJsonlDataset(
        images_root=args.train_images,
        gold_jsonl=args.golden_gold_jsonl,
        transform=build_transforms(train=False),
    )

    # Test_2 requires data_reliance.
    # Therefore require_reliance=True.
    mixed_train_ds = CifarJsonlRelianceDataset(
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

    gold_train_loader = DataLoader(
        gold_train_ds,
        batch_size=args.gold_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    gold_eval_loader = DataLoader(
        gold_eval_ds,
        batch_size=args.gold_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    gold_stability_loader = DataLoader(
        gold_eval_ds,
        batch_size=args.gold_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    mixed_train_loader = DataLoader(
        mixed_train_ds,
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

    print(f"[INFO] device={device}")
    print(f"[INFO] n_gold={len(gold_train_ds)}")
    print(f"[INFO] n_mixed_train_without_gold={len(mixed_train_ds)}")
    print(f"[INFO] n_test={len(test_ds)}")
    print(f"[INFO] reliance_jsonl={args.reliance_jsonl}")
    print(f"[INFO] train_filter_noisy={args.train_filter_noisy}")
    print(f"[INFO] result_dir={result_dir}")
    print(f"[INFO] save_dir={save_dir}")
    print(f"[INFO] metrics_path={metrics_path}")
    print(f"[INFO] summary_path={summary_path}")
    print(f"[INFO] config_path={config_path}")

    # -------------------------------------------------------------
    # Stage 1: train on gold data to obtain theta_G
    # -------------------------------------------------------------
    learner = SimpleCNN(num_classes=10).to(device)

    train_gold_reference(
        model=learner,
        gold_loader=gold_train_loader,
        test_loader=test_loader,
        device=device,
        args=args,
    )

    torch.save(
        {
            "model": learner.state_dict(),
            "args": vars(args),
            "stage": "gold_reference",
        },
        gold_ckpt,
    )

    print(f"[DONE] gold reference saved: {gold_ckpt}")

    # -------------------------------------------------------------
    # Stage 2:
    # learner starts from theta_G.
    # frozen evaluator is a copy of theta_G.
    # -------------------------------------------------------------
    gold_evaluator = copy.deepcopy(learner).to(device)
    gold_evaluator.eval()

    for p in gold_evaluator.parameters():
        p.requires_grad_(False)

    print("[STAGE 2] Computing gold prototypes")

    prototypes = compute_gold_prototypes(
        gold_model=gold_evaluator,
        gold_loader=gold_eval_loader,
        device=device,
        num_classes=10,
    )

    print(f"[DONE] prototypes shape={tuple(prototypes.shape)}")

    gold_hook = FeatureHook(gold_evaluator)

    optimizer = torch.optim.SGD(
        learner.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    ce_none = nn.CrossEntropyLoss(reduction="none")
    ce_mean = nn.CrossEntropyLoss()

    steps_per_epoch = len(mixed_train_loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    global_step = 0

    best_test_acc = -1.0
    best_epoch = -1

    gold_iter = iter(gold_stability_loader)

    print("[STAGE 2] Hybrid FME critical learning on mixed data")
    print(
        f"[INFO] epochs={args.epochs} base_lr={args.base_lr} min_lr={args.min_lr} "
        f"scheduler={args.scheduler}"
    )
    print(
        f"[INFO] beta={args.beta} gamma_gold={args.gamma_gold} "
        f"tau={args.tau} alpha={args.alpha} "
        f"lambda_g={args.lambda_g} anti_weight={args.anti_weight} "
        f"grad_clip={args.grad_clip}"
    )

    # -------------------------------------------------------------
    # Mixed-data training
    # -------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        learner.train()

        run_total_loss = 0.0
        run_mixed_loss = 0.0
        run_positive_loss = 0.0
        run_anti_loss = 0.0
        run_gold_loss = 0.0
        run_acc = 0.0

        run_r = 0.0
        run_r_gold = 0.0
        run_r_llm = 0.0
        run_r_prob = 0.0
        run_r_sim = 0.0

        run_c = 0.0
        run_c_pos_mean = 0.0
        run_c_neg_mean = 0.0

        run_c_pos_frac = 0.0
        run_c_zero_frac = 0.0
        run_c_neg_frac = 0.0

        n_seen = 0

        for x, y, data_reliance in mixed_train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            data_reliance = data_reliance.to(device, non_blocking=True).float()

            try:
                xg, yg = next(gold_iter)
            except StopIteration:
                gold_iter = iter(gold_stability_loader)
                xg, yg = next(gold_iter)

            xg = xg.to(device, non_blocking=True)
            yg = yg.to(device, non_blocking=True)

            if args.scheduler == "cosine":
                lr_t = cosine_lr(global_step, total_steps, args.base_lr, args.min_lr)
            else:
                lr_t = args.base_lr

            optimizer.param_groups[0]["lr"] = lr_t

            # -----------------------------------------------------
            # Hybrid reliability score:
            #
            # r_gold = beta * r_prob + (1 - beta) * r_sim
            # r_llm  = data_reliance
            # r      = gamma_gold * r_gold + (1 - gamma_gold) * r_llm
            #
            # For this first Test_2 run:
            # gamma_gold = 0.5, so gold evaluator and data reliance are balanced.
            # -----------------------------------------------------
            with torch.no_grad():
                r_gold, r_prob, r_sim = compute_gold_consistency(
                    gold_model=gold_evaluator,
                    gold_hook=gold_hook,
                    prototypes=prototypes,
                    x=x,
                    y=y,
                    beta=args.beta,
                )

                r_llm = data_reliance.clamp(0.0, 1.0)

                r = args.gamma_gold * r_gold + (1.0 - args.gamma_gold) * r_llm
                r = r.clamp(0.0, 1.0)

                c = controller_from_reliability(
                    r=r,
                    alpha=args.alpha,
                    tau=args.tau,
                )

            optimizer.zero_grad(set_to_none=True)

            logits = learner(x)

            mixed_loss, positive_loss, anti_fit_loss, c_pos, c_neg = stable_controlled_losses(
                logits=logits,
                y=y,
                c=c,
                ce_none=ce_none,
                anti_weight=args.anti_weight,
            )

            logits_g_current = learner(xg)
            gold_stability_loss = ce_mean(logits_g_current, yg)

            total_loss = mixed_loss + args.lambda_g * gold_stability_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("[ERROR] total_loss became NaN/Inf.")
                print(f"mixed_loss={mixed_loss.item()}")
                print(f"positive_loss={positive_loss.item()}")
                print(f"anti_fit_loss={anti_fit_loss.item()}")
                print(f"gold_stability_loss={gold_stability_loss.item()}")
                print(
                    f"r_mean={r.mean().item()} "
                    f"r_gold={r_gold.mean().item()} "
                    f"r_llm={r_llm.mean().item()} "
                    f"c_mean={c.mean().item()}"
                )
                raise FloatingPointError("NaN/Inf detected in total_loss.")

            total_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(learner.parameters(), args.grad_clip)

            optimizer.step()

            bs = y.size(0)
            n_seen += bs

            run_total_loss += total_loss.item() * bs
            run_mixed_loss += mixed_loss.item() * bs
            run_positive_loss += positive_loss.item() * bs
            run_anti_loss += anti_fit_loss.item() * bs
            run_gold_loss += gold_stability_loss.item() * bs
            run_acc += accuracy(logits, y) * bs

            run_r += r.mean().item() * bs
            run_r_gold += r_gold.mean().item() * bs
            run_r_llm += r_llm.mean().item() * bs
            run_r_prob += r_prob.mean().item() * bs
            run_r_sim += r_sim.mean().item() * bs

            run_c += c.mean().item() * bs
            run_c_pos_mean += c_pos.mean().item() * bs
            run_c_neg_mean += c_neg.mean().item() * bs

            run_c_pos_frac += (c > 0.05).float().mean().item() * bs
            run_c_zero_frac += ((c >= -0.05) & (c <= 0.05)).float().mean().item() * bs
            run_c_neg_frac += (c < -0.05).float().mean().item() * bs

            global_step += 1

        train_total_loss = run_total_loss / max(1, n_seen)
        train_mixed_loss = run_mixed_loss / max(1, n_seen)
        train_positive_loss = run_positive_loss / max(1, n_seen)
        train_anti_loss = run_anti_loss / max(1, n_seen)
        train_gold_loss = run_gold_loss / max(1, n_seen)
        train_acc = run_acc / max(1, n_seen)

        mean_r = run_r / max(1, n_seen)
        mean_r_gold = run_r_gold / max(1, n_seen)
        mean_r_llm = run_r_llm / max(1, n_seen)
        mean_r_prob = run_r_prob / max(1, n_seen)
        mean_r_sim = run_r_sim / max(1, n_seen)

        mean_c = run_c / max(1, n_seen)
        mean_c_pos = run_c_pos_mean / max(1, n_seen)
        mean_c_neg = run_c_neg_mean / max(1, n_seen)

        frac_c_pos = run_c_pos_frac / max(1, n_seen)
        frac_c_zero = run_c_zero_frac / max(1, n_seen)
        frac_c_neg = run_c_neg_frac / max(1, n_seen)

        test_loss, test_acc = evaluate(learner, test_loader, device)

        dt = time.time() - t0
        lr_last = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={lr_last:.6f} | "
            f"total_loss={train_total_loss:.4f} "
            f"mixed_loss={train_mixed_loss:.4f} "
            f"pos_loss={train_positive_loss:.4f} "
            f"anti_loss={train_anti_loss:.4f} "
            f"gold_loss={train_gold_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"r={mean_r:.4f} "
            f"r_gold={mean_r_gold:.4f} "
            f"r_llm={mean_r_llm:.4f} "
            f"r_prob={mean_r_prob:.4f} "
            f"r_sim={mean_r_sim:.4f} "
            f"c={mean_c:.4f} "
            f"c_pos={mean_c_pos:.4f} "
            f"c_neg={mean_c_neg:.4f} "
            f"c+={frac_c_pos:.3f} "
            f"c0={frac_c_zero:.3f} "
            f"c-={frac_c_neg:.3f} | "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f} | "
            f"time={dt:.1f}s"
        )

        row = {
            "epoch": epoch,
            "lr_last": lr_last,
            "scheduler": "FME_Evaluator_Test_2_hybrid_gold_data_reliance",
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "gold_epochs": args.gold_epochs,
            "gold_lr": args.gold_lr,
            "beta": args.beta,
            "gamma_gold": args.gamma_gold,
            "tau": args.tau,
            "alpha": args.alpha,
            "lambda_g": args.lambda_g,
            "anti_weight": args.anti_weight,
            "grad_clip": args.grad_clip,
            "train_total_loss": train_total_loss,
            "train_mixed_loss": train_mixed_loss,
            "train_positive_loss": train_positive_loss,
            "train_anti_fit_loss": train_anti_loss,
            "train_gold_stability_loss": train_gold_loss,
            "train_acc_on_observed_mixed_labels": train_acc,
            "mean_r": mean_r,
            "mean_r_gold": mean_r_gold,
            "mean_r_llm_data_reliance": mean_r_llm,
            "mean_r_prob": mean_r_prob,
            "mean_r_sim": mean_r_sim,
            "mean_c": mean_c,
            "mean_c_pos": mean_c_pos,
            "mean_c_neg": mean_c_neg,
            "frac_c_positive": frac_c_pos,
            "frac_c_near_zero": frac_c_zero,
            "frac_c_negative": frac_c_neg,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "time_sec": dt,
            "n_gold": len(gold_train_ds),
            "n_mixed_train_without_gold": len(mixed_train_ds),
            "n_test": len(test_ds),
            "golden_gold_jsonl": args.golden_gold_jsonl,
            "train_jsonl": args.train_jsonl,
            "test_jsonl": args.test_jsonl,
            "train_images": args.train_images,
            "test_images": args.test_images,
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
                    "model": learner.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "best_test_acc": best_test_acc,
                    "best_epoch": best_epoch,
                },
                best_ckpt,
            )

    gold_hook.close()

    torch.save(
        {
            "epoch": args.epochs,
            "model": learner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
        },
        final_ckpt,
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "FME_Evaluator_Test_2_hybrid_gold_data_reliance",
                "best_test_acc": best_test_acc,
                "best_epoch": best_epoch,
                "best_ckpt": str(best_ckpt),
                "final_ckpt": str(final_ckpt),
                "gold_reference_ckpt": str(gold_ckpt),
                "metrics_jsonl": str(metrics_path),
                "summary_json": str(summary_path),
                "config_json": str(config_path),
                "result_dir": str(result_dir),
                "save_dir": str(save_dir),
                "n_gold": len(gold_train_ds),
                "n_mixed_train_without_gold": len(mixed_train_ds),
                "n_test": len(test_ds),
                "golden_gold_jsonl": args.golden_gold_jsonl,
                "train_jsonl": args.train_jsonl,
                "test_jsonl": args.test_jsonl,
                "reliance_jsonl": args.reliance_jsonl,
                "beta": args.beta,
                "gamma_gold": args.gamma_gold,
                "tau": args.tau,
                "alpha": args.alpha,
                "lambda_g": args.lambda_g,
                "anti_weight": args.anti_weight,
                "grad_clip": args.grad_clip,
                "gold_epochs": args.gold_epochs,
                "epochs": args.epochs,
                "base_lr": args.base_lr,
                "min_lr": args.min_lr,
                "gold_lr": args.gold_lr,
                "train_filter_noisy": args.train_filter_noisy,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    print(f"[DONE] best_test_acc={best_test_acc:.4f} epoch={best_epoch}")
    print(f"[DONE] best checkpoint saved: {best_ckpt}")
    print(f"[DONE] final checkpoint saved: {final_ckpt}")
    print(f"[DONE] metrics saved: {metrics_path}")
    print(f"[DONE] summary saved: {summary_path}")


if __name__ == "__main__":
    main()