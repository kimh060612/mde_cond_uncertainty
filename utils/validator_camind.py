from typing import Dict

import torch
from torch.utils.data import DataLoader
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


def _new_accumulator() -> dict:
    return {
        "loss": 0.0,
        "nll_loss": 0.0,
        "mean_loss": 0.0,
        "variance_loss": 0.0,
        "ranking_loss": 0.0,
        "processed_batches": 0,
        "q_rank_accuracy": 0.0,
        "q_rank_accuracy_count": 0,
        "vectors": {
            "target_ssi_loss": [],
            "camera_bias": [],
            "sigma": [],
            "q_score": [],
            "candidate_abs_rel": [],
            "canonical_abs_rel": [],
            "abs_rel_degradation": [],
        },
    }


def _append_vectors(
    accumulator: dict,
    sample_mask: torch.Tensor | None = None,
    **items: torch.Tensor,
) -> None:
    for key, value in items.items():
        value = value.detach().float().flatten()
        if sample_mask is not None:
            value = value[sample_mask]
        if value.numel() > 0:
            accumulator["vectors"][key].append(value.cpu())


def _add_rank_accuracy(
    accumulator: dict,
    rank_accuracy: float,
) -> None:
    if torch.isfinite(torch.tensor(rank_accuracy)):
        accumulator["q_rank_accuracy"] += rank_accuracy
        accumulator["q_rank_accuracy_count"] += 1


def _finalize_accumulator(
    accumulator: dict,
    max_samples: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    processed_batches = accumulator["processed_batches"]
    if processed_batches > 0:
        for key in ("loss", "nll_loss", "mean_loss", "variance_loss", "ranking_loss"):
            metrics[key] = accumulator[key] / processed_batches

    rank_count = accumulator["q_rank_accuracy_count"]
    if rank_count > 0:
        metrics["q_rank_accuracy"] = accumulator["q_rank_accuracy"] / rank_count

    cat_vectors = {
        key: torch.cat(values, dim=0) if values else None
        for key, values in accumulator["vectors"].items()
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


@torch.no_grad()
def validate(
    epoch: int,
    model_id: str,
    model,
    loader: DataLoader,
    device,
    amp: bool,
    lambda_smooth_logvar: float,
    lambda_variance: float,
    listnet_temperature: float,
    uncertainty_mode: str,
    list_loss_weight: float,
    seen_topology_numbers: torch.Tensor = None,
    unseen_topology_numbers: torch.Tensor = None,
    correlation_max_samples: int = 100_000,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    relative_align_mode: str = "scale_shift",
    uncertainty_alpha: float = 1.0,
):
    del model_id, lambda_smooth_logvar, uncertainty_mode, min_depth, max_depth, relative_align_mode

    model.eval()
    total_accumulator = _new_accumulator()
    seen_accumulator = _new_accumulator()
    unseen_accumulator = _new_accumulator()

    progress_bar = tqdm(
        loader,
        desc=f"Validation {epoch:03d}",
        dynamic_ncols=True,
        leave=False,
    )

    for step, batch in enumerate(progress_bar, start=1):
        if batch is None:
            continue

        num_groups, num_candidates = batch["candidate_images"].shape[:2]
        flat_batch = tensor_device(flatten_group_batch(batch), device)
        candidate_imgs = flat_batch["candidate_images"]
        canonical_imgs = flat_batch["canonical_images"]
        camera_context = flat_batch["camera_context"]
        abs_rel_degradation = flat_batch["abs_rel_degradation"]

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
            group_q = reshape_group_batch(q_score, num_groups, num_candidates)
            group_degradation = reshape_group_batch(
                abs_rel_degradation,
                num_groups,
                num_candidates,
            )
            ranking_loss = signed_pairwise_ranknet_loss(
                group_q,
                group_degradation,
                temperature=listnet_temperature,
            )
            nll_loss = mean_loss + lambda_variance * variance_loss
            loss = nll_loss + list_loss_weight * ranking_loss

        batch_vectors = {
            "target_ssi_loss": target_loss,
            "camera_bias": out["camera_bias"],
            "sigma": out["std"],
            "q_score": q_score,
            "candidate_abs_rel": flat_batch["candidate_abs_rel"],
            "canonical_abs_rel": flat_batch["canonical_abs_rel"],
            "abs_rel_degradation": abs_rel_degradation,
        }
        rank_accuracy = _pairwise_rank_accuracy(group_q, group_degradation)

        total_accumulator["loss"] += float(loss.item())
        total_accumulator["nll_loss"] += float(nll_loss.item())
        total_accumulator["mean_loss"] += float(mean_loss.item())
        total_accumulator["variance_loss"] += float(variance_loss.item())
        total_accumulator["ranking_loss"] += float(ranking_loss.item())
        total_accumulator["processed_batches"] += 1
        _add_rank_accuracy(total_accumulator, rank_accuracy)
        _append_vectors(total_accumulator, **batch_vectors)

        group_topology = batch["info"][:, 6].to(device=device).long()
        if seen_topology_numbers is not None:
            seen_group_mask = torch.isin(
                group_topology,
                seen_topology_numbers.to(device=device).long(),
            )
            seen_sample_mask = seen_group_mask[:, None].expand(-1, num_candidates).reshape(-1)
            _append_vectors(seen_accumulator, seen_sample_mask, **batch_vectors)
            if seen_group_mask.any():
                _add_rank_accuracy(
                    seen_accumulator,
                    _pairwise_rank_accuracy(
                        group_q[seen_group_mask],
                        group_degradation[seen_group_mask],
                    ),
                )
        if unseen_topology_numbers is not None:
            unseen_group_mask = torch.isin(
                group_topology,
                unseen_topology_numbers.to(device=device).long(),
            )
            unseen_sample_mask = unseen_group_mask[:, None].expand(-1, num_candidates).reshape(-1)
            _append_vectors(unseen_accumulator, unseen_sample_mask, **batch_vectors)
            if unseen_group_mask.any():
                _add_rank_accuracy(
                    unseen_accumulator,
                    _pairwise_rank_accuracy(
                        group_q[unseen_group_mask],
                        group_degradation[unseen_group_mask],
                    ),
                )

        n = max(total_accumulator["processed_batches"], 1)
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_accumulator['loss'] / n:.4f}",
            ssi=f"{_finite_mean(target_loss):.4f}",
            deg=f"{_finite_mean(abs_rel_degradation):.4f}",
            q_acc=f"{total_accumulator['q_rank_accuracy'] / max(total_accumulator['q_rank_accuracy_count'], 1):.4f}",
        )

    total_metrics = _finalize_accumulator(total_accumulator, correlation_max_samples)
    seen_metrics = _finalize_accumulator(seen_accumulator, correlation_max_samples)
    unseen_metrics = _finalize_accumulator(unseen_accumulator, correlation_max_samples)

    return total_metrics, seen_metrics, unseen_metrics
