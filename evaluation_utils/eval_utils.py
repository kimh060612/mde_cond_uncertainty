from collections.abc import Sequence
from typing import Dict, Tuple, Optional
import math
import torch

from evaluation_utils.eval_selection import (
    DEFAULT_RELATIVE_REGRET_THRESHOLDS,
    compute_selection_metrics,
)


DEGRADATION_EPS = 1e-6


def depth_error_maps(
    mu: torch.Tensor,
    target: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: Optional[float] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = ensure_bchw(mu)
    target = ensure_bchw(target)
    if max_depth is None:
        mu = mu.clamp_min(min_depth)
        target = target.clamp_min(min_depth)
    else:
        mu = torch.clamp(mu, min_depth, max_depth)
        target = torch.clamp(target, min_depth, max_depth)

    abs_rel_error = torch.abs(mu - target) / (target + eps)
    ratio = torch.maximum(mu / (target + eps), target / (mu + eps))
    a1_error = (ratio >= 1.25).float()

    return abs_rel_error, a1_error

def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected [B, H, W] or [B, 1, H, W], got {tuple(x.shape)}")

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_counts = mask.flatten(1).sum(dim=1)
    sums = torch.where(mask, values, torch.zeros_like(values)).flatten(1).sum(dim=1)
    means = sums / valid_counts.clamp_min(1).to(dtype=values.dtype)
    return torch.where(valid_counts > 0, means, torch.full_like(means, float("nan")))

def masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_values = values.flatten(1)
    flat_mask = mask.flatten(1)
    valid_counts = flat_mask.sum(dim=1)
    sorted_values = torch.where(
        flat_mask,
        flat_values,
        torch.full_like(flat_values, float("inf")),
    ).sort(dim=1).values
    median_idx = ((valid_counts - 1).clamp_min(0) // 2).unsqueeze(1)
    medians = sorted_values.gather(1, median_idx).squeeze(1)
    return torch.where(valid_counts > 0, medians, torch.full_like(medians, float("nan")))


@torch.no_grad()
def compute_relative_alignment(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    align_mode: str = "scale_shift",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = ensure_bchw(pred)
    gt = ensure_bchw(gt)
    valid_mask = ensure_bchw(valid_mask).bool()
    calc_dtype = torch.float64 if pred.dtype == torch.float64 or gt.dtype == torch.float64 else torch.float32
    pred = pred.to(dtype=calc_dtype)
    gt = gt.to(dtype=calc_dtype)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: target {gt.shape}, pred {pred.shape}")
    if valid_mask.shape != pred.shape:
        valid_mask = valid_mask.expand_as(pred)

    relative_mask = valid_mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0) & (pred > 0)
    gt_inv = 1.0 / (gt + eps)

    if align_mode == "median":
        pred_median = masked_median(pred, relative_mask)
        gt_inv_median = masked_median(gt_inv, relative_mask)
        scale = gt_inv_median / (pred_median + eps)
        shift = torch.zeros_like(scale)
    elif align_mode == "scale_shift":
        x = torch.where(relative_mask, pred, torch.zeros_like(pred))
        y = torch.where(relative_mask, gt_inv, torch.zeros_like(gt_inv))

        valid_counts = relative_mask.flatten(1).sum(dim=1).to(dtype=calc_dtype)
        safe_counts = valid_counts.clamp_min(1.0)
        sum_x = x.flatten(1).sum(dim=1)
        sum_y = y.flatten(1).sum(dim=1)
        sum_xx = (x * x).flatten(1).sum(dim=1)
        sum_xy = (x * y).flatten(1).sum(dim=1)

        denom = safe_counts * sum_xx - sum_x.square()
        stable = (valid_counts > 1) & (denom.abs() > eps)
        scale = torch.where(
            stable,
            (safe_counts * sum_xy - sum_x * sum_y) / denom.clamp(min=eps),
            torch.zeros_like(valid_counts),
        )
        shift = torch.where(
            valid_counts > 0,
            (sum_y - scale * sum_x) / safe_counts,
            torch.zeros_like(valid_counts),
        )
    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")

    return scale.view(-1, 1, 1, 1), shift.view(-1, 1, 1, 1)


def align_relative_prediction_to_depth_space(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    log_var: Optional[torch.Tensor] = None,
    align_mode: str = "scale_shift",
    inv_depth_min: float = 1e-1,
    eps: float = 1e-8,
) -> Dict[str, Optional[torch.Tensor]]:
    pred = ensure_bchw(pred)
    gt = ensure_bchw(gt)
    calc_dtype = torch.float64 if pred.dtype == torch.float64 or gt.dtype == torch.float64 else torch.float32
    pred = pred.to(dtype=calc_dtype)

    scale, shift = compute_relative_alignment(
        pred,
        gt,
        valid_mask,
        align_mode=align_mode,
        eps=eps,
    )
    aligned_inv_depth = scale * pred + shift
    safe_inv_depth = aligned_inv_depth.clamp_min(inv_depth_min)
    pred_depth = 1.0 / safe_inv_depth

    result = {
        "depth": pred_depth,
        "aligned_inv_depth": safe_inv_depth,
        "scale": scale,
        "shift": shift,
        "std": None,
        "log_var": None,
    }

    scale_abs = scale.abs().clamp_min(eps)
    if sigma is not None:
        sigma_inv = ensure_bchw(sigma).to(dtype=calc_dtype) * scale_abs
        result["std"] = sigma_inv / safe_inv_depth.square()

    if log_var is not None:
        log_var_inv = ensure_bchw(log_var).to(dtype=calc_dtype) + 2.0 * torch.log(scale_abs)
        result["log_var"] = log_var_inv - 4.0 * torch.log(safe_inv_depth)

    return result

def metric_dict(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    min_depth=1e-3,
    max_depth=80.0,
    eps=1e-8,
    calc_dtype=torch.float32,
) -> Dict[str, torch.Tensor]:
    pred_depth = torch.clamp(pred_depth, min_depth, max_depth)
    gt_depth = torch.clamp(gt_depth, min_depth, max_depth)
    mask = mask & torch.isfinite(pred_depth) & torch.isfinite(gt_depth)

    diff = gt_depth - pred_depth
    sq_error = diff.square()
    thresh = torch.maximum(gt_depth / (pred_depth + eps), pred_depth / (gt_depth + eps))

    return {
        "abs_rel": masked_mean(torch.abs(diff) / (gt_depth + eps), mask),
        "rmse": torch.sqrt(masked_mean(sq_error, mask)),
        "a1": masked_mean((thresh < 1.25).to(dtype=calc_dtype), mask),
        "a2": masked_mean((thresh < 1.25 ** 2).to(dtype=calc_dtype), mask),
        "a3": masked_mean((thresh < 1.25 ** 3).to(dtype=calc_dtype), mask),
    }

@torch.no_grad()
def align_relative_depth_and_uncertainty(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    align_mode: str = "scale_shift",
    calc_dtype=torch.float32,
):
    aligned = align_relative_prediction_to_depth_space(
        pred,
        gt,
        valid_mask,
        sigma=sigma,
        align_mode=align_mode,
    )
    if sigma is not None:
        return aligned["depth"], aligned["std"]
    return aligned["depth"]

def _accumulate_finite_metrics(metric_sums, metric_counts, metrics):
    for key, value in metrics.items():
        if key.endswith("_samples"):
            continue
        value = float(value)
        if math.isfinite(value):
            metric_sums[key] = metric_sums.get(key, 0.0) + value
            metric_counts[key] = metric_counts.get(key, 0) + 1


def _mean_finite_metrics(metric_sums, metric_counts):
    return {
        key: metric_sums[key] / metric_counts[key]
        for key in metric_sums
        if metric_counts.get(key, 0) > 0
    }


def finite_mean(values: torch.Tensor) -> float:
    values = values.detach().float().flatten()
    finite_mask = torch.isfinite(values)
    if not finite_mask.any():
        return float("nan")
    return float(values[finite_mask].mean().item())


def pairwise_rank_counts(
    predicted_score: torch.Tensor,
    target_score: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[int, int]:
    _, num_candidates = predicted_score.shape
    if num_candidates < 2:
        return 0, 0

    pair_i, pair_j = torch.triu_indices(
        num_candidates,
        num_candidates,
        offset=1,
        device=predicted_score.device,
    )
    pred_diff = predicted_score[:, pair_i] - predicted_score[:, pair_j]
    target_diff = target_score[:, pair_i] - target_score[:, pair_j]
    pair_valid_mask = (
        torch.isfinite(pred_diff)
        & torch.isfinite(target_diff)
        & (target_diff != 0)
    )
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if valid_mask.shape != predicted_score.shape:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} does not match "
                f"predicted_score shape {tuple(predicted_score.shape)}"
            )
        pair_valid_mask &= valid_mask[:, pair_i] & valid_mask[:, pair_j]
    if not pair_valid_mask.any():
        return 0, 0

    correct = (
        torch.sign(pred_diff[pair_valid_mask])
        == torch.sign(target_diff[pair_valid_mask])
    )
    return int(correct.sum().item()), int(pair_valid_mask.sum().item())


def abs_rel_degradation_targets(
    predicted_score: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    canonical_abs_rel: torch.Tensor,
    eps: float = DEGRADATION_EPS,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build AbsRel degradation targets with one shared validity mask."""
    predicted_score = predicted_score.detach().float().flatten()
    candidate_abs_rel = candidate_abs_rel.detach().float().flatten()
    canonical_abs_rel = canonical_abs_rel.detach().float().flatten()
    if not (
        predicted_score.shape
        == candidate_abs_rel.shape
        == canonical_abs_rel.shape
    ):
        raise ValueError(
            "predicted_score, candidate_abs_rel, and canonical_abs_rel must "
            "have the same flattened shape"
        )

    valid_mask = (
        torch.isfinite(predicted_score)
        & torch.isfinite(candidate_abs_rel)
        & torch.isfinite(canonical_abs_rel)
        & (candidate_abs_rel >= 0)
        & (canonical_abs_rel >= 0)
    )
    absolute = candidate_abs_rel - canonical_abs_rel
    percentage = absolute / (canonical_abs_rel + eps) * 100.0
    log_ratio = (
        torch.log(candidate_abs_rel + eps)
        - torch.log(canonical_abs_rel + eps)
    )
    valid_mask &= (
        torch.isfinite(absolute)
        & torch.isfinite(percentage)
        & torch.isfinite(log_ratio)
    )
    return {
        "abs_rel_degradation": absolute,
        "abs_rel_degradation_percent": percentage,
        "abs_rel_degradation_log": log_ratio,
    }, valid_mask


def degradation_correlation_metrics(
    predicted_score: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    canonical_abs_rel: torch.Tensor,
    max_samples: int,
    score_name: str,
) -> Dict[str, float]:
    from evaluation_utils.eval_metrics import (
        compute_vector_masked_correlations,
    )

    targets, valid_mask = abs_rel_degradation_targets(
        predicted_score,
        candidate_abs_rel,
        canonical_abs_rel,
    )
    metrics: Dict[str, float] = {}
    for target_name, target in targets.items():
        metrics.update(
            compute_vector_masked_correlations(
                target,
                predicted_score,
                valid_mask=valid_mask,
                prefix=f"{score_name}_vs_{target_name}",
                max_samples=max_samples,
            )
        )
    return metrics


def new_validation_accumulator() -> dict:
    return {
        "loss": 0.0,
        "nll_loss": 0.0,
        "mean_loss": 0.0,
        "variance_loss": 0.0,
        "ranking_loss": 0.0,
        "processed_batches": 0,
        "q_rank_correct": 0,
        "q_rank_total": 0,
        "vectors": {
            "target_ssi_loss": [],
            "camera_bias": [],
            "sigma": [],
            "q_score": [],
            "candidate_abs_rel": [],
            "canonical_abs_rel": [],
            "abs_rel_degradation": [],
            "rmse_degradation": [],
            "group_id": [],
        },
    }


def append_accumulator_vectors(
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


def add_rank_counts(
    accumulator: dict,
    rank_counts: tuple[int, int],
) -> None:
    correct, total = rank_counts
    accumulator["q_rank_correct"] += correct
    accumulator["q_rank_total"] += total


def concatenate_accumulator_vectors(
    accumulator: dict,
) -> dict[str, torch.Tensor | None]:
    return {
        key: torch.cat(values, dim=0) if values else None
        for key, values in accumulator["vectors"].items()
    }


def finalize_validation_accumulator(
    accumulator: dict,
    max_samples: int,
    *,
    selection_min_settings: int = 10,
    selection_thresholds: Sequence[
        float
    ] = DEFAULT_RELATIVE_REGRET_THRESHOLDS,
    concatenated_vectors: dict[str, torch.Tensor | None] | None = None,
) -> Dict[str, float]:
    from evaluation_utils.eval_metrics import (
        compute_vector_masked_correlations,
    )

    metrics: Dict[str, float] = {}
    processed_batches = accumulator["processed_batches"]
    if processed_batches > 0:
        for key in (
            "loss",
            "nll_loss",
            "mean_loss",
            "variance_loss",
            "ranking_loss",
        ):
            metrics[key] = accumulator[key] / processed_batches

    rank_total = accumulator["q_rank_total"]
    if rank_total > 0:
        metrics["q_rank_accuracy"] = (
            accumulator["q_rank_correct"] / rank_total
        )

    vectors = (
        concatenate_accumulator_vectors(accumulator)
        if concatenated_vectors is None
        else concatenated_vectors
    )
    for key, value in vectors.items():
        if value is not None and key not in {"rmse_degradation", "group_id"}:
            metrics[key] = finite_mean(value)

    target_loss = vectors.get("target_ssi_loss")
    camera_bias = vectors.get("camera_bias")
    rmse_degradation = vectors.get("rmse_degradation")
    q_score = vectors.get("q_score")
    candidate_abs_rel = vectors.get("candidate_abs_rel")
    canonical_abs_rel = vectors.get("canonical_abs_rel")
    group_id = vectors.get("group_id")

    if target_loss is not None and camera_bias is not None:
        metrics.update(
            compute_vector_masked_correlations(
                target_loss,
                camera_bias,
                prefix="bias_vs_ssi_loss",
                max_samples=max_samples,
            )
        )
    if (
        camera_bias is not None
        and candidate_abs_rel is not None
        and canonical_abs_rel is not None
    ):
        metrics.update(
            degradation_correlation_metrics(
                camera_bias,
                candidate_abs_rel,
                canonical_abs_rel,
                max_samples,
                score_name="bias",
            )
        )
    if (
        q_score is not None
        and candidate_abs_rel is not None
        and canonical_abs_rel is not None
    ):
        metrics.update(
            degradation_correlation_metrics(
                q_score,
                candidate_abs_rel,
                canonical_abs_rel,
                max_samples,
                score_name="q",
            )
        )
        if group_id is not None:
            selection_metrics = compute_selection_metrics(
                q_score,
                candidate_abs_rel,
                group_id,
                min_settings_per_group=selection_min_settings,
                relative_regret_thresholds=selection_thresholds,
            )
            metrics.update(
                {
                    key: value
                    for key, value in selection_metrics.items()
                    if key != "num_groups"
                }
            )
    if rmse_degradation is not None and q_score is not None:
        rmse_valid_mask = (
            torch.isfinite(rmse_degradation)
            & (rmse_degradation != -1.0)
        )
        if candidate_abs_rel is not None and canonical_abs_rel is not None:
            _, abs_rel_valid_mask = abs_rel_degradation_targets(
                q_score,
                candidate_abs_rel,
                canonical_abs_rel,
            )
            rmse_valid_mask &= abs_rel_valid_mask
        metrics.update(
            compute_vector_masked_correlations(
                rmse_degradation,
                q_score,
                valid_mask=rmse_valid_mask,
                prefix="q_vs_rmse_degradation",
                max_samples=max_samples,
            )
        )
    return metrics
