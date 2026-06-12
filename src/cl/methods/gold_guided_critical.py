from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class GoldScoreBatch:
    r_prob: torch.Tensor
    r_sim: torch.Tensor
    gold_consistency: torch.Tensor
    control: torch.Tensor


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def freeze_batch_norm_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


@torch.no_grad()
def compute_gold_prototypes(
    evaluator: nn.Module,
    gold_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    evaluator.eval()
    prototype_sums = None
    class_counts = torch.zeros(num_classes, device=device)

    for images, labels in tqdm(gold_loader, desc="Build gold prototypes", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = evaluator.forward_features(images)

        if prototype_sums is None:
            prototype_sums = torch.zeros(
                num_classes,
                features.size(1),
                device=device,
                dtype=features.dtype,
            )

        prototype_sums.index_add_(0, labels, features)
        class_counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))

    if prototype_sums is None:
        raise ValueError("Cannot build prototypes from an empty gold loader.")

    missing_classes = torch.where(class_counts == 0)[0].cpu().tolist()
    if missing_classes:
        raise ValueError(f"Gold data is missing labels: {missing_classes}")

    prototypes = prototype_sums / class_counts.unsqueeze(1)
    return prototypes.detach(), class_counts.detach()


@torch.no_grad()
def compute_gold_scores(
    evaluator: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    beta: float,
    alpha: float,
    tau: float,
    controller_mode: str,
    min_control: float,
) -> GoldScoreBatch:
    evaluator.eval()
    logits = evaluator(images)
    probabilities = torch.softmax(logits, dim=1)
    r_prob = probabilities.gather(1, labels.view(-1, 1)).squeeze(1)

    features = evaluator.forward_features(images)
    label_prototypes = prototypes[labels]
    cosine_similarity = F.cosine_similarity(features, label_prototypes, dim=1)
    r_sim = (cosine_similarity + 1.0) / 2.0

    gold_consistency = beta * r_prob + (1.0 - beta) * r_sim
    raw_control = torch.tanh(alpha * (gold_consistency - tau))
    # This implementation intentionally uses non-negative sample weights:
    # unreliable samples are down-weighted or skipped, not anti-fitted.
    if controller_mode == "positive":
        positive_control = (raw_control + 1.0) / 2.0
        control = min_control + (1.0 - min_control) * positive_control
    elif controller_mode == "clamped":
        control = raw_control.clamp_min(min_control)
    else:
        raise ValueError(f"Unknown controller_mode: {controller_mode}")
    return GoldScoreBatch(
        r_prob=r_prob,
        r_sim=r_sim,
        gold_consistency=gold_consistency,
        control=control,
    )


def cycle_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    stage_name: str,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"{stage_name} epoch {epoch}", leave=False)
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


def train_critical_epoch(
    learner: nn.Module,
    evaluator: nn.Module,
    mixed_loader: DataLoader,
    gold_batch_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
    prototypes: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    beta: float,
    alpha: float,
    tau: float,
    lambda_gold: float,
    controller_mode: str,
    min_control: float,
    grad_clip: float | None,
    freeze_learner_batch_norm: bool,
) -> dict[str, float]:
    learner.train()
    if freeze_learner_batch_norm:
        freeze_batch_norm_stats(learner)
    evaluator.eval()
    total_examples = 0
    total_loss = 0.0
    total_mixed_loss = 0.0
    total_stability_loss = 0.0
    total_correct = 0
    total_r_prob = 0.0
    total_r_sim = 0.0
    total_gold_consistency = 0.0
    total_control = 0.0

    progress = tqdm(mixed_loader, desc=f"Critical epoch {epoch}", leave=False)
    for mixed_images, mixed_labels in progress:
        gold_images, gold_labels = next(gold_batch_iterator)
        mixed_images = mixed_images.to(device, non_blocking=True)
        mixed_labels = mixed_labels.to(device, non_blocking=True)
        gold_images = gold_images.to(device, non_blocking=True)
        gold_labels = gold_labels.to(device, non_blocking=True)

        gold_scores = compute_gold_scores(
            evaluator=evaluator,
            images=mixed_images,
            labels=mixed_labels,
            prototypes=prototypes,
            beta=beta,
            alpha=alpha,
            tau=tau,
            controller_mode=controller_mode,
            min_control=min_control,
        )

        optimizer.zero_grad(set_to_none=True)
        mixed_logits = learner(mixed_images)
        per_sample_losses = F.cross_entropy(
            mixed_logits,
            mixed_labels,
            reduction="none",
        )
        mixed_loss = (gold_scores.control * per_sample_losses).mean()

        gold_logits = learner(gold_images)
        stability_loss = F.cross_entropy(gold_logits, gold_labels)
        loss = mixed_loss + lambda_gold * stability_loss
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(learner.parameters(), grad_clip)
        optimizer.step()

        batch_size = mixed_labels.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_mixed_loss += mixed_loss.item() * batch_size
        total_stability_loss += stability_loss.item() * batch_size
        total_correct += (mixed_logits.argmax(dim=1) == mixed_labels).sum().item()
        total_r_prob += gold_scores.r_prob.sum().item()
        total_r_sim += gold_scores.r_sim.sum().item()
        total_gold_consistency += gold_scores.gold_consistency.sum().item()
        total_control += gold_scores.control.sum().item()

        progress.set_postfix(
            loss=total_loss / total_examples,
            acc=total_correct / total_examples,
            gold_consistency=total_gold_consistency / total_examples,
            control=total_control / total_examples,
        )

    return {
        "train_loss": total_loss / total_examples,
        "mixed_loss": total_mixed_loss / total_examples,
        "gold_stability_loss": total_stability_loss / total_examples,
        "train_accuracy": total_correct / total_examples,
        "mean_r_prob": total_r_prob / total_examples,
        "mean_r_sim": total_r_sim / total_examples,
        "mean_gold_consistency": total_gold_consistency / total_examples,
        "mean_control": total_control / total_examples,
    }
