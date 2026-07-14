from typing import Dict, Tuple
import logging

import torch
from tqdm.auto import tqdm

from dataset.ati_dataset_caminduce import flatten_group_batch
from evaluation_utils.eval_metrics import compute_vector_masked_correlations
from model.loss_fn import (
    scalar_heteroscedastic_loss,
    scale_shift_invariant_depth_loss,
    signed_pairwise_ranknet_loss,
)
from utils.train_utils import reshape_group_batch, tensor_device


def _finite_mean(values: torch.Tensor) -> float:
    values = values.detach().float().flatten()
    finite_mask = torch.isfinite(values)
    if not finite_mask.any():
        return float("nan")
    return float(values[finite_mask].mean().item())


def _pairwise_rank_accuracy(
    predicted_score: torch.Tensor,
    target_score: torch.Tensor,
) -> float:
    _, num_candidates = predicted_score.shape
    if num_candidates < 2:
        return float("nan")

    pair_i, pair_j = torch.triu_indices(
        num_candidates,
        num_candidates,
        offset=1,
        device=predicted_score.device,
    )
    pred_diff = predicted_score[:, pair_i] - predicted_score[:, pair_j]
    target_diff = target_score[:, pair_i] - target_score[:, pair_j]
    valid_mask = torch.isfinite(pred_diff) & torch.isfinite(target_diff) & (target_diff != 0)
    if not valid_mask.any():
        return float("nan")
    correct = torch.sign(pred_diff[valid_mask]) == torch.sign(target_diff[valid_mask])
    return float(correct.float().mean().item())


def _cat(values: list[torch.Tensor]) -> torch.Tensor | None:
    if not values:
        return None
    return torch.cat(values, dim=0)


def _append_epoch_vectors(
    vectors: dict[str, list[torch.Tensor]],
    **items: torch.Tensor,
) -> None:
    for key, value in items.items():
        vectors[key].append(value.detach().float().flatten().cpu())


def _summarize_epoch_vectors(
    vectors: dict[str, list[torch.Tensor]],
    max_samples: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    cat_vectors = {
        key: _cat(value)
        for key, value in vectors.items()
    }

    for key, value in cat_vectors.items():
        if value is not None:
            metrics[key] = _finite_mean(value)

    target_loss = cat_vectors.get("target_ssi_loss")
    camera_bias = cat_vectors.get("camera_bias")
    abs_rel_degradation = cat_vectors.get("abs_rel_degradation")
    q_score = cat_vectors.get("q_score")

    if target_loss is not None and camera_bias is not None:
        metrics.update(
            compute_vector_masked_correlations(
                target_loss,
                camera_bias,
                prefix="bias_vs_ssi_loss",
                max_samples=max_samples,
            )
        )
    if abs_rel_degradation is not None and camera_bias is not None:
        metrics.update(
            compute_vector_masked_correlations(
                abs_rel_degradation,
                camera_bias,
                prefix="bias_vs_abs_rel_degradation",
                max_samples=max_samples,
            )
        )
    if abs_rel_degradation is not None and q_score is not None:
        metrics.update(
            compute_vector_masked_correlations(
                abs_rel_degradation,
                q_score,
                prefix="q_vs_abs_rel_degradation",
                max_samples=max_samples,
            )
        )
    return metrics


def train_one_epoch(
    model_id: str,
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    amp: bool,
    lambda_smooth_logvar: float,
    lambda_variance: float,
    list_loss_weight: float,
    listnet_temperature: float,
    uncertainty_mode: str,
    grad_clip: float,
    logger: logging.Logger,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    correlation_max_samples: int = 100_000,
    relative_align_mode: str = "scale_shift",
    uncertainty_alpha: float = 1.0,
    global_step: int = 0,
    log_interval: int = 20,
) -> Tuple[Dict[str, float], int]:
    del model_id, lambda_smooth_logvar, uncertainty_mode, min_depth, max_depth

    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)

    model.train()
    progress_bar = tqdm(
        loader,
        desc=f"Train {epoch:03d}",
        dynamic_ncols=True,
        leave=False,
    )

    running = {
        "loss": 0.0,
        "nll_loss": 0.0,
        "mean_loss": 0.0,
        "variance_loss": 0.0,
        "ranking_loss": 0.0,
        "q_rank_accuracy": 0.0,
    }
    rank_accuracy_count = 0
    processed_batches = 0
    vectors: dict[str, list[torch.Tensor]] = {
        "target_ssi_loss": [],
        "camera_bias": [],
        "sigma": [],
        "q_score": [],
        "candidate_abs_rel": [],
        "canonical_abs_rel": [],
        "abs_rel_degradation": [],
    }

    for step, batch in enumerate(progress_bar, start=1):
        if batch is None:
            continue

        num_groups, num_candidates = batch["candidate_images"].shape[:2]
        flat_batch = tensor_device(flatten_group_batch(batch), device)
        candidate_imgs = flat_batch["candidate_images"]
        canonical_imgs = flat_batch["canonical_images"]
        camera_context = flat_batch["camera_context"]
        abs_rel_degradation = flat_batch["abs_rel_degradation"]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(
                candidate_imgs,
                canonical_imgs,
                camera_context,
                target_size=candidate_imgs.shape[-2:],
            )
            target_loss = scale_shift_invariant_depth_loss(
                out["candidate_depth"],
                out["canonical_depth"],
            )
            mean_loss, variance_loss = scalar_heteroscedastic_loss(
                out["camera_bias"],
                out["variance"],
                target_loss,
            )
            q_score = out["camera_bias"] # + uncertainty_alpha * out["std"]
            ranking_loss = signed_pairwise_ranknet_loss(
                reshape_group_batch(q_score, num_groups, num_candidates),
                reshape_group_batch(target_loss, num_groups, num_candidates), # abs_rel_degradation
                temperature=listnet_temperature,
            )
            nll_loss = mean_loss + lambda_variance * variance_loss
            loss = nll_loss + list_loss_weight * ranking_loss

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            group_q = reshape_group_batch(q_score.detach(), num_groups, num_candidates)
            group_degradation = reshape_group_batch(
                target_loss.detach(),  # abs_rel_degradation
                num_groups,
                num_candidates,
            )
            rank_accuracy = _pairwise_rank_accuracy(group_q, group_degradation)
            if torch.isfinite(torch.tensor(rank_accuracy)):
                running["q_rank_accuracy"] += rank_accuracy
                rank_accuracy_count += 1

            _append_epoch_vectors(
                vectors,
                target_ssi_loss=target_loss,
                camera_bias=out["camera_bias"],
                sigma=out["std"],
                q_score=q_score,
                candidate_abs_rel=flat_batch["candidate_abs_rel"],
                canonical_abs_rel=flat_batch["canonical_abs_rel"],
                abs_rel_degradation=abs_rel_degradation,
            )

        running["loss"] += float(loss.item())
        running["nll_loss"] += float(nll_loss.item())
        running["mean_loss"] += float(mean_loss.item())
        running["variance_loss"] += float(variance_loss.item())
        running["ranking_loss"] += float(ranking_loss.item())
        processed_batches += 1
        global_step += 1

        n = max(processed_batches, 1)
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{running['loss'] / n:.4f}",
            ssi=f"{_finite_mean(target_loss):.4f}",
            deg=f"{_finite_mean(abs_rel_degradation):.4f}",
            q_acc=f"{running['q_rank_accuracy'] / max(rank_accuracy_count, 1):.4f}",
        )

        if log_interval > 0 and step % log_interval == 0:
            logger.info(
                "epoch=%d step=%d/%d avg_loss=%.6f mean_loss=%.6f variance_loss=%.6f ranking_loss=%.6f",
                epoch,
                step,
                len(loader),
                running["loss"] / n,
                running["mean_loss"] / n,
                running["variance_loss"] / n,
                running["ranking_loss"] / n,
            )

    n = max(processed_batches, 1)
    epoch_metrics = {
        key: value / n
        for key, value in running.items()
        if key != "q_rank_accuracy"
    }
    epoch_metrics["q_rank_accuracy"] = running["q_rank_accuracy"] / max(rank_accuracy_count, 1)
    epoch_metrics.update(_summarize_epoch_vectors(vectors, correlation_max_samples))
    return epoch_metrics, global_step
