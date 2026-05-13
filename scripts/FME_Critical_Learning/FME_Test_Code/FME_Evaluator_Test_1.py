import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Import shared project files
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()

CANDIDATE_IMPORT_DIRS = [
    THIS_FILE.parent,
    THIS_FILE.parent.parent,
    Path("/users/zeyuhan/charlie_codebase/AI-Critical-Learning/scripts/CIFAR-10/Experiment_and_Test/Improved_Method/Function_4"),
    Path("/users/zeyuhan/charlie_codebase/AI-Critical-Learning/scripts/FME_Critical_Learning"),
]

for p in CANDIDATE_IMPORT_DIRS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_reliance import (
    CifarJsonlRelianceDataset,
    build_transforms,
    read_jsonl,
    resolve_image_path,
)
from utils import set_seed, accuracy, cosine_lr, evaluate
from model import SimpleCNN


# ---------------------------------------------------------------------
# Gold dataset
# ---------------------------------------------------------------------
@dataclass
class GoldSample:
    raw_id: int
    relpath: str
    label: int
    class_name: str
    meta: Dict[str, Any]


class GoldJsonlDataset(Dataset):
    """
    Dataset for Golden_Gold_Standard_seed42.jsonl.

    Expected row examples may include:
      {"gg_id": 0, "raw_id": 5736, ...}
      {"raw_id": 5736, "label": 3, "relpath": "images/005736.png", ...}

    This class supports:
      - raw_id or id
      - label, y, class_id, or orig_label
      - relpath if available
      - fallback path: images/{raw_id:06d}.png
    """

    def __init__(
        self,
        images_root: str,
        gold_jsonl: str,
        transform=None,
    ):
        self.images_root = Path(images_root)
        self.gold_jsonl = gold_jsonl
        self.transform = transform

        rows = read_jsonl(gold_jsonl)
        samples: List[GoldSample] = []

        for obj in rows:
            if "raw_id" in obj:
                raw_id = int(obj["raw_id"])
            elif "id" in obj:
                raw_id = int(obj["id"])
            else:
                raise KeyError(f"Gold JSONL row has no raw_id or id: {obj}")

            if "label" in obj:
                label = int(obj["label"])
            elif "y" in obj:
                label = int(obj["y"])
            elif "class_id" in obj:
                label = int(obj["class_id"])
            elif "orig_label" in obj:
                label = int(obj["orig_label"])
            else:
                raise KeyError(f"Gold JSONL row has no label/y/class_id/orig_label: {obj}")

            if "relpath" in obj:
                relpath = str(obj["relpath"])
            else:
                relpath = f"images/{raw_id:06d}.png"

            class_name = str(obj.get("class_name", obj.get("orig_class", "")))

            samples.append(
                GoldSample(
                    raw_id=raw_id,
                    relpath=relpath,
                    label=label,
                    class_name=class_name,
                    meta=obj,
                )
            )

        if len(samples) == 0:
            raise RuntimeError(f"Gold dataset is empty: {gold_jsonl}")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_path = resolve_image_path(self.images_root, s.relpath)

        if not img_path.exists():
            fallback = self.images_root / f"{s.raw_id:06d}.png"
            if fallback.exists():
                img_path = fallback
            else:
                raise FileNotFoundError(
                    f"Gold image not found. Tried: {img_path} and {fallback}. "
                    f"images_root={self.images_root}, relpath={s.relpath}, raw_id={s.raw_id}"
                )

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, s.label


# ---------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------
class FeatureHook:
    """
    Extracts the feature vector before the final Linear classifier.

    It finds the last nn.Linear layer in SimpleCNN and captures its input.
    That input is treated as h(x).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.features: Optional[torch.Tensor] = None
        self.handle = None

        last_linear = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is None:
            raise RuntimeError(
                "Could not find nn.Linear layer in model. "
                "Please check SimpleCNN architecture."
            )

        self.handle = last_linear.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        feat = inputs[0]
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        self.features = feat

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.features = None
        logits = self.model(x)

        if self.features is None:
            raise RuntimeError("Feature hook did not capture features.")

        return logits, self.features

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ---------------------------------------------------------------------
# Gold-guided components
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_gold_prototypes(
    gold_model: nn.Module,
    gold_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> torch.Tensor:
    """
    Compute class prototypes:

        mu_y^G = average h_{theta_G}(x_j^G), for y_j^G = y

    Returns:
        prototypes: Tensor [num_classes, feature_dim]
    """
    gold_model.eval()
    hook = FeatureHook(gold_model)

    sums = None
    counts = torch.zeros(num_classes, device=device)

    for x, y in gold_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        _, feat = hook.get_features(x)
        feat = feat.detach()

        if sums is None:
            sums = torch.zeros(num_classes, feat.size(1), device=device)

        for cls in range(num_classes):
            mask = y == cls
            if mask.any():
                sums[cls] += feat[mask].sum(dim=0)
                counts[cls] += mask.sum()

    hook.close()

    if sums is None:
        raise RuntimeError("Gold loader is empty. Cannot compute prototypes.")

    missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()
    if len(missing) > 0:
        print(f"[WARN] Missing gold prototypes for classes: {missing}. Their prototypes will be zeros.")

    prototypes = sums / counts.clamp_min(1.0).unsqueeze(1)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    return prototypes


@torch.no_grad()
def compute_gold_consistency(
    gold_model: nn.Module,
    gold_hook: FeatureHook,
    prototypes: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute:

        r_prob_i = p_{theta_G}(y_i | x_i)

        r_sim_i = (cos(h_{theta_G}(x_i), mu^G_{y_i}) + 1) / 2

        r_i = beta * r_prob_i + (1 - beta) * r_sim_i
    """
    gold_model.eval()

    logits_g, feat_g = gold_hook.get_features(x)
    prob_g = F.softmax(logits_g, dim=1)

    r_prob = prob_g.gather(1, y.view(-1, 1)).squeeze(1)

    feat_g = F.normalize(feat_g, p=2, dim=1)
    proto_y = prototypes[y]

    cos_sim = (feat_g * proto_y).sum(dim=1)
    r_sim = (cos_sim + 1.0) / 2.0
    r_sim = r_sim.clamp(0.0, 1.0)

    r = beta * r_prob + (1.0 - beta) * r_sim
    r = r.clamp(0.0, 1.0)

    return r, r_prob, r_sim


def controller_from_reliability(
    r: torch.Tensor,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    """
    Elastic critical controller:

        c_i = tanh(alpha * (r_i - tau))
    """
    return torch.tanh(alpha * (r - tau))


def stable_controlled_losses(
    logits: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    ce_none: nn.Module,
    anti_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stable implementation of evaluator-centered control.

    Important:
    Do NOT directly optimize mean(c_i * CE_i) when c_i can be negative.
    Negative CE is unbounded and can cause logits to explode into NaN.

    Instead:
      c_i > 0:
          weighted CE learning

      c_i < 0:
          stable anti-fitting loss:
              -log(1 - p_theta(y_i | x_i))

          This pushes down the probability of the observed unreliable label
          without using an unbounded negative CE objective.
    """
    loss_each = ce_none(logits, y)

    c_detached = c.detach()
    c_pos = torch.clamp(c_detached, min=0.0)
    c_neg = torch.clamp(-c_detached, min=0.0)

    positive_loss = (c_pos * loss_each).mean()

    prob = F.softmax(logits, dim=1)
    p_y = prob.gather(1, y.view(-1, 1)).squeeze(1)

    anti_fit_loss_each = -torch.log((1.0 - p_y).clamp_min(1e-6))
    anti_fit_loss = (c_neg * anti_fit_loss_each).mean()

    mixed_loss = positive_loss + anti_weight * anti_fit_loss

    return mixed_loss, positive_loss, anti_fit_loss, c_pos, c_neg


# ---------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------
def train_gold_reference(
    model: nn.Module,
    gold_loader: DataLoader,
    test_loader: Optional[DataLoader],
    device: torch.device,
    args,
) -> None:
    """
    Stage 1:
        Train model on gold data to obtain theta_G.
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.gold_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    ce = nn.CrossEntropyLoss()

    steps_per_epoch = len(gold_loader)
    total_steps = max(1, args.gold_epochs * steps_per_epoch)
    global_step = 0

    print("[STAGE 1] Gold reference training")
    print(f"[INFO] n_gold={len(gold_loader.dataset)}")
    print(f"[INFO] gold_epochs={args.gold_epochs} gold_lr={args.gold_lr} gold_min_lr={args.gold_min_lr}")

    for epoch in range(1, args.gold_epochs + 1):
        t0 = time.time()
        model.train()

        run_loss = 0.0
        run_acc = 0.0
        n_seen = 0

        for x, y in gold_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if args.gold_scheduler == "cosine":
                lr_t = cosine_lr(global_step, total_steps, args.gold_lr, args.gold_min_lr)
            else:
                lr_t = args.gold_lr

            optimizer.param_groups[0]["lr"] = lr_t

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            run_loss += loss.item() * bs
            run_acc += accuracy(logits, y) * bs
            n_seen += bs
            global_step += 1

        gold_train_loss = run_loss / max(1, n_seen)
        gold_train_acc = run_acc / max(1, n_seen)
        dt = time.time() - t0

        msg = (
            f"[GOLD] Epoch {epoch:03d}/{args.gold_epochs} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"gold_loss={gold_train_loss:.4f} gold_acc={gold_train_acc:.4f} | "
            f"time={dt:.1f}s"
        )

        if test_loader is not None and args.eval_during_gold:
            test_loss, test_acc = evaluate(model, test_loader, device)
            msg += f" | test_loss={test_loss:.4f} test_acc={test_acc:.4f}"

        print(msg)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("FME Evaluator Test 1: Gold-Guided Critical Learning")

    # Data paths
    p.add_argument("--train_images", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--test_images", type=str, required=True)
    p.add_argument("--test_jsonl", type=str, required=True)

    # Kept for compatibility with old commands.
    # This new algorithm does not use old reliance score as controller.
    p.add_argument("--reliance_jsonl", type=str, default=None)

    # Gold data
    p.add_argument("--golden_gold_jsonl", type=str, required=True)

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gold_batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)

    # Gold reference training
    p.add_argument("--gold_epochs", type=int, default=5)
    p.add_argument("--gold_lr", type=float, default=0.05)
    p.add_argument("--gold_min_lr", type=float, default=0.001)
    p.add_argument("--gold_scheduler", type=str, default="cosine", choices=["constant", "cosine"])
    p.add_argument("--eval_during_gold", action="store_true")

    # Mixed training LR
    p.add_argument("--base_lr", type=float, default=0.03)
    p.add_argument("--min_lr", type=float, default=0.001)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["constant", "cosine"])

    # Optimizer
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # Gold-guided critical controller
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--lambda_g", type=float, default=0.5)
    p.add_argument("--anti_weight", type=float, default=0.1)

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
        save_dir = result_dir / "runs" / "FME_Evaluator_Test_1"
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

    mixed_train_ds = CifarJsonlRelianceDataset(
        images_root=args.train_images,
        index_jsonl=args.train_jsonl,
        split="train",
        transform=build_transforms(train=True),
        reliance_jsonl=args.reliance_jsonl,
        require_reliance=False,
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

    print("[STAGE 2] Gold-guided critical learning on mixed data")
    print(
        f"[INFO] epochs={args.epochs} base_lr={args.base_lr} min_lr={args.min_lr} "
        f"scheduler={args.scheduler}"
    )
    print(
        f"[INFO] beta={args.beta} tau={args.tau} alpha={args.alpha} "
        f"lambda_g={args.lambda_g} anti_weight={args.anti_weight} grad_clip={args.grad_clip}"
    )

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
        run_r_prob = 0.0
        run_r_sim = 0.0
        run_c = 0.0
        run_c_pos_mean = 0.0
        run_c_neg_mean = 0.0

        run_c_pos_frac = 0.0
        run_c_zero_frac = 0.0
        run_c_neg_frac = 0.0

        n_seen = 0

        for x, y, _old_reliance in mixed_train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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

            with torch.no_grad():
                r, r_prob, r_sim = compute_gold_consistency(
                    gold_model=gold_evaluator,
                    gold_hook=gold_hook,
                    prototypes=prototypes,
                    x=x,
                    y=y,
                    beta=args.beta,
                )

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
                print(f"r_mean={r.mean().item()} c_mean={c.mean().item()}")
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
            f"r={mean_r:.4f} r_prob={mean_r_prob:.4f} r_sim={mean_r_sim:.4f} "
            f"c={mean_c:.4f} c_pos={mean_c_pos:.4f} c_neg={mean_c_neg:.4f} "
            f"c+={frac_c_pos:.3f} c0={frac_c_zero:.3f} c-={frac_c_neg:.3f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"time={dt:.1f}s"
        )

        row = {
            "epoch": epoch,
            "lr_last": lr_last,
            "scheduler": "FME_Evaluator_Test_1",
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "gold_epochs": args.gold_epochs,
            "gold_lr": args.gold_lr,
            "beta": args.beta,
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
            "reliance_jsonl_compat_only": args.reliance_jsonl,
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
                "method": "FME_Evaluator_Test_1",
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
                "beta": args.beta,
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