import math
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


def cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    if total_steps <= 1:
        return min_lr
    t = step / (total_steps - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def scale_from_reliance(r_bar: float, args) -> float:
    r_bar = float(max(0.0, min(1.0, r_bar)))
    if args.map_type == "piecewise":
        if r_bar >= args.thr_high:
            return args.scale_high
        if r_bar >= args.thr_mid:
            return args.scale_mid
        return args.scale_low
    # exp
    s = math.exp(-args.k * (1.0 - r_bar))
    s = max(args.scale_min, min(args.scale_max, s))
    return float(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y, _r in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += ce_sum(logits, y).item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += y.numel()
    return total_loss / total_n, total_correct / total_n


def compute_lr(mode: str, step: int, total_steps: int, base_lr: float, min_lr: float, scale: float) -> float:
    """
    mode:
      - cosine_only: lr = cosine(step)
      - reliance_only: lr = base_lr * scale
      - full: lr = cosine(step) * scale
    """
    if mode == "cosine_only":
        return cosine_lr(step, total_steps, base_lr, min_lr)
    if mode == "reliance_only":
        return base_lr * scale
    if mode == "full":
        return cosine_lr(step, total_steps, base_lr, min_lr) * scale
    raise ValueError(f"Unknown ablation_mode: {mode}")