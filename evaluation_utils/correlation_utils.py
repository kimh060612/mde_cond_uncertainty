"""
Evaluate correlation between per-pixel loss and uncertainty, and compute sparsification AUSE metrics.
- Evaluation in Training Loop
    - AUSE for AbsRel / A1 and Uncertainty
    - AURG for AbsRel / A1 and Uncertainty

- Evaluation in Validation Loop
    - AUSE for AbsRel / A1 and Uncertainty
    - AURG for AbsRel / A1 and Uncertainty
    - ARU, RMSU metric for image-level mean uncertainty
    - Per-pixel Correlation between L1 depth loss and uncertainty (Pearson, Spearman)
    - Global Correlations between image-level mean uncertainty and image-level depth metrics (AbsRel, A1)

When calculating AUSE, AURG, ARU, RMSU metrics, apply median scaling for relative depth models.
In metric depth estimation models, no scaling is applied. The metrics are computed per-image and then averaged across the batch.
"""


from typing import Dict, Optional, Tuple
from evaluation_utils.eval_utils import _masked_median
import torch

### AUSE, AURG metric
@torch.no_grad()
def compute_sparsification_ause_metrics(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    max_samples: Optional[int] = 100_000,
    num_bins: int = 100,
    model_type: str = "relative",
) -> Dict[str, torch.Tensor]:
    """
    Compute MDE-style AUSE metrics by removing high-uncertainty pixels first.

    ``ause_abs_rel`` uses per-pixel AbsRel error. ``ause_a1`` uses the a1
    failure indicator, so lower AUSE is better for both metrics.
    """
    abs_rel_error, a1_error = _depth_error_maps(mu, target, model_type, valid_mask)
    uncertainty = _ensure_bchw(uncertainty)
    if uncertainty.shape != abs_rel_error.shape:
        uncertainty = uncertainty.expand_as(abs_rel_error)

    mask = _prepare_bchw_mask(valid_mask, abs_rel_error)
    finite_mask = (
        mask
        & torch.isfinite(abs_rel_error)
        & torch.isfinite(a1_error)
        & torch.isfinite(uncertainty)
    )

    samples = _prepare_sparsification_samples(
        abs_rel_error,
        a1_error,
        uncertainty,
        finite_mask,
        max_samples=max_samples,
    )
    if samples is None:
        empty = abs_rel_error.new_full((abs_rel_error.shape[0],), float("nan"), dtype=torch.float64)
        return {
            "ause_abs_rel": empty,
            "ause_a1": empty,
            "ause_samples": 0,
        }
    abs_rel_samples, a1_samples, uncertainty_samples, sample_mask = samples

    ause_abs_rel = _batched_sparsification_ause(
        abs_rel_samples,
        uncertainty_samples,
        sample_mask,
        num_bins=num_bins,
    )
    ause_a1 = _batched_sparsification_ause(
        a1_samples,
        uncertainty_samples,
        sample_mask,
        num_bins=num_bins,
    )
    return {
        "ause_abs_rel": ause_abs_rel,
        "ause_a1": ause_a1
    }

@torch.no_grad()
def compute_sparsification_aurg_metrics(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    max_samples: Optional[int] = 100_000,
    num_bins: int = 100,
    model_type: str = "relative",
) -> Dict[str, torch.Tensor]:
    """
    Compute MDE-style AURG metrics by removing high-uncertainty pixels first.

    ``aurg_abs_rel`` uses per-pixel AbsRel error. ``aurg_a1`` uses the a1
    failure indicator, so higher AURG is better for both metrics.
    """
    abs_rel_error, a1_error = _depth_error_maps(mu, target, model_type, valid_mask)
    uncertainty = _ensure_bchw(uncertainty)
    if uncertainty.shape != abs_rel_error.shape:
        uncertainty = uncertainty.expand_as(abs_rel_error)

    mask = _prepare_bchw_mask(valid_mask, abs_rel_error)
    finite_mask = (
        mask
        & torch.isfinite(abs_rel_error)
        & torch.isfinite(a1_error)
        & torch.isfinite(uncertainty)
    )

    samples = _prepare_sparsification_samples(
        abs_rel_error,
        a1_error,
        uncertainty,
        finite_mask,
        max_samples=max_samples,
    )
    if samples is None:
        empty = abs_rel_error.new_full((abs_rel_error.shape[0],), float("nan"), dtype=torch.float64)
        return {
            "aurg_abs_rel": empty,
            "aurg_a1": empty,
            "aurg_samples": 0,
        }
    abs_rel_samples, a1_samples, uncertainty_samples, sample_mask = samples

    aurg_abs_rel = _batched_sparsification_aurg(
        abs_rel_samples,
        uncertainty_samples,
        sample_mask,
        num_bins=num_bins,
    )
    aurg_a1 = _batched_sparsification_aurg(
        a1_samples,
        uncertainty_samples,
        sample_mask,
        num_bins=num_bins,
    )
    return {
        "aurg_abs_rel": aurg_abs_rel,
        "aurg_a1": aurg_a1,
    }


### ARU, RMSU metric
@torch.no_grad()
def compute_aru_rmsu_metrics(
    mu: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    model_type: str = "relative",
) -> Dict[str, torch.Tensor]:
    gt = _ensure_bchw(gt)
    valid_mask = _prepare_bchw_mask(valid_mask, gt)
    l1_loss_map = calculate_l1_depth_lossmap(mu, gt, valid_mask, model_type)
    uncertainty = _ensure_bchw(uncertainty)
    if uncertainty.shape != l1_loss_map.shape:
        uncertainty = uncertainty.expand_as(l1_loss_map)

    finite_mask = (
        valid_mask
        & torch.isfinite(gt)
        & torch.isfinite(l1_loss_map)
        & torch.isfinite(uncertainty)
    )
    counts = finite_mask.flatten(1).sum(dim=1)
    uncertainty_error = uncertainty - l1_loss_map
    aru_values = torch.abs(uncertainty_error / gt.clamp_min(1e-3))
    rmsu_values = uncertainty_error.square()
    aru_sum = torch.where(finite_mask, aru_values, torch.zeros_like(aru_values)).flatten(1).sum(dim=1)
    rmsu_sum = torch.where(finite_mask, rmsu_values, torch.zeros_like(rmsu_values)).flatten(1).sum(dim=1)
    safe_counts = counts.clamp_min(1).to(dtype=aru_values.dtype)
    aru_map = aru_sum / safe_counts
    rmsu_map = torch.sqrt(rmsu_sum / safe_counts)
    aru_map = torch.where(counts > 0, aru_map, aru_map.new_full(aru_map.shape, float("nan")))
    rmsu_map = torch.where(counts > 0, rmsu_map, rmsu_map.new_full(rmsu_map.shape, float("nan")))

    return {
        "aru": aru_map,
        "rmsu": rmsu_map
    }

### Global Correlation Metrics
@torch.no_grad()
def compute_masked_correlations(
    x: torch.Tensor,
    y: torch.Tensor,
    valid_mask: torch.Tensor,
    max_samples: Optional[int] = 100_000,
    prefix: str = "correlation",
) -> Dict[str, float]:
    """
    Compute Pearson and Spearman correlations between two per-pixel maps.

    The valid pixels are optionally sub-sampled with deterministic uniform
    indexing so this metric does not perturb the training RNG state.
    """
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x {tuple(x.shape)} != y {tuple(y.shape)}")

    x_flat, y_flat = _prepare_masked_vectors(x, y, valid_mask, max_samples=max_samples)
    if x_flat.numel() < 2:
        return {
            f"{prefix}_pearson": float("nan"),
            f"{prefix}_spearman": float("nan"),
        }

    pearson = _pearson_corr(x_flat, y_flat)
    spearman = _spearman_corr(x_flat, y_flat)

    return {
        f"{prefix}_pearson": float(pearson.item()),
        f"{prefix}_spearman": float(spearman.item()),
    }

@torch.no_grad()
def compute_loss_uncertainty_correlations(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: Optional[torch.Tensor] = None,
    max_samples: Optional[int] = 100_000,
    model_type="relative"
) -> Dict[str, torch.Tensor]:
    """
    Compute correlation between per-pixel L1 depth loss and uncertainty.

    Args:
        uncertainty_kind:
            Used only when ``uncertainty`` is None. One of "std", "var", or
            "log_var".
    """
    loss_map = calculate_l1_depth_lossmap(mu, target, valid_mask, model_type=model_type)
    if uncertainty is None:
        uncertainty = torch.exp(0.5 * log_var)
    uncertainty = _ensure_bchw(uncertainty)
    if uncertainty.shape != loss_map.shape:
        uncertainty = uncertainty.expand_as(loss_map)

    mask = _prepare_bchw_mask(valid_mask, loss_map)
    finite_mask = mask & torch.isfinite(loss_map) & torch.isfinite(uncertainty)
    samples = _prepare_sparsification_samples(
        loss_map,
        loss_map,
        uncertainty,
        finite_mask,
        max_samples=max_samples,
    )
    if samples is None:
        empty = loss_map.new_full((loss_map.shape[0],), float("nan"), dtype=torch.float64)
        return {
            "loss_uncertainty_pearson": empty,
            "loss_uncertainty_spearman": empty,
        }

    loss_samples, _, uncertainty_samples, sample_mask = samples
    return {
        "loss_uncertainty_pearson": _batched_pearson_corr(
            loss_samples,
            uncertainty_samples,
            sample_mask,
        ),
        "loss_uncertainty_spearman": _batched_spearman_corr(
            loss_samples,
            uncertainty_samples,
            sample_mask,
        ),
    }


### Utility functions for internal use

def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected tensor with shape [B, H, W] or [B, 1, H, W], got {tuple(x.shape)}")

def _prepare_bchw_mask(
    valid_mask: torch.Tensor, 
    reference: torch.Tensor
) -> torch.Tensor:
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)

    mask = valid_mask.bool()
    if mask.shape != reference.shape:
        mask = mask.expand_as(reference)

    return mask


def _deterministic_subsample(
    *vectors: torch.Tensor,
    max_samples: Optional[int],
) -> Tuple[torch.Tensor, ...]:
    if not vectors:
        return ()

    numel = vectors[0].numel()
    if max_samples is None or max_samples <= 0 or numel <= max_samples:
        return vectors

    if max_samples == 1:
        idx = torch.zeros(1, device=vectors[0].device, dtype=torch.long)
    else:
        idx = torch.arange(max_samples, device=vectors[0].device, dtype=torch.long)
        idx = idx * (numel - 1) // (max_samples - 1)
    return tuple(v[idx] for v in vectors)

def _depth_error_maps(
    mu: torch.Tensor,
    target: torch.Tensor,
    model_type: str = "relative",
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = _ensure_bchw(mu).clamp_min(1e-3)
    target = _ensure_bchw(target).clamp_min(1e-3)
    if valid_mask is not None:
        valid_mask = _prepare_bchw_mask(valid_mask, target)
    eps = 1e-8

    if model_type == "metric":
        abs_rel_error = torch.abs(mu - target) / target
        ratio = torch.maximum(mu / target, target / mu)
        a1_error = (ratio >= 1.25).float()
    else:
        gt_inv = 1.0 / (target + 1e-6)
        relative_mask = (target > 0) & (mu > 0)
        if valid_mask is not None:
            relative_mask = relative_mask & valid_mask

        pred_median = _masked_median(mu, relative_mask)
        gt_inv_median = _masked_median(gt_inv, relative_mask)
        scale = gt_inv_median / (pred_median + eps)
        pred_aligned = mu * scale.view(-1, 1, 1, 1)
        
        abs_rel_error = torch.abs(pred_aligned - target) / target
        ratio = torch.maximum(pred_aligned / target, target / pred_aligned)
        a1_error = (ratio >= 1.25).float()
    return abs_rel_error, a1_error

def calculate_l1_depth_lossmap(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor = None,
    model_type: str = "relative",
) -> torch.Tensor:
    mu = _ensure_bchw(mu)
    target = _ensure_bchw(target)
    if valid_mask is None:
        valid_mask = torch.ones_like(target, dtype=torch.bool)
    else:
        valid_mask = _prepare_bchw_mask(valid_mask, target)
    if model_type == "metric":
        l1_loss_map = torch.abs(mu - target)
    else:
        eps = 1e-8
        gt_inv = 1.0 / (target + 1e-6)
        relative_mask = (target > 0) & (mu > 0) & valid_mask

        pred_median = _masked_median(mu, relative_mask)
        gt_inv_median = _masked_median(gt_inv, relative_mask)
        scale = gt_inv_median / (pred_median + eps)
        pred_aligned = mu * scale.view(-1, 1, 1, 1)
        l1_loss_map = torch.abs(pred_aligned - target)
    return l1_loss_map

def _prepare_masked_vectors(
    x: torch.Tensor,
    y: torch.Tensor,
    valid_mask: torch.Tensor,
    max_samples: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = _prepare_bchw_mask(valid_mask, x)

    finite_mask = mask & torch.isfinite(x) & torch.isfinite(y)
    x = x[finite_mask].detach().flatten()
    y = y[finite_mask].detach().flatten()

    x, y = _deterministic_subsample(x, y, max_samples=max_samples)

    return x.float(), y.float()


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x.numel() < 2:
        return x.new_tensor(float("nan"))

    x = x.double()
    y = y.double()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.square().sum() * y_centered.square().sum()).clamp_min(eps))

    if denom <= eps:
        return x.new_tensor(float("nan"))

    return (x_centered * y_centered).sum() / denom


def _ordinal_ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float64)
    return ranks


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2:
        return x.new_tensor(float("nan"))

    return _pearson_corr(_ordinal_ranks(x), _ordinal_ranks(y))

def _batched_pearson_corr(
    x: torch.Tensor,
    y: torch.Tensor,
    sample_mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    batch_size = x.shape[0]
    result = x.new_full((batch_size,), float("nan"), dtype=torch.float64)

    x = x.double()
    y = y.double()
    sample_mask = sample_mask.bool()
    counts = sample_mask.sum(dim=1)
    valid_rows = counts >= 2
    if not valid_rows.any():
        return result

    safe_x = torch.where(sample_mask, x, torch.zeros_like(x))
    safe_y = torch.where(sample_mask, y, torch.zeros_like(y))
    safe_counts = counts.clamp_min(1).to(dtype=torch.float64).unsqueeze(1)
    x_centered = torch.where(sample_mask, x - safe_x.sum(dim=1, keepdim=True) / safe_counts, torch.zeros_like(x))
    y_centered = torch.where(sample_mask, y - safe_y.sum(dim=1, keepdim=True) / safe_counts, torch.zeros_like(y))
    denom = torch.sqrt(x_centered.square().sum(dim=1) * y_centered.square().sum(dim=1))
    corr = (x_centered * y_centered).sum(dim=1) / denom.clamp_min(eps)
    valid_rows = valid_rows & (denom > eps)
    return torch.where(valid_rows, corr, result)


def _batched_ordinal_ranks(x: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    x = x.double()
    sample_mask = sample_mask.bool()
    order = torch.argsort(
        torch.where(sample_mask, x, torch.full_like(x, float("inf"))),
        dim=1,
    )
    ranks = torch.empty_like(x, dtype=torch.float64)
    rank_values = torch.arange(x.shape[1], device=x.device, dtype=torch.float64).expand_as(x)
    ranks.scatter_(1, order, rank_values)
    return torch.where(sample_mask, ranks, torch.zeros_like(ranks))


def _batched_spearman_corr(
    x: torch.Tensor,
    y: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    return _batched_pearson_corr(
        _batched_ordinal_ranks(x, sample_mask),
        _batched_ordinal_ranks(y, sample_mask),
        sample_mask,
    )

def _prepare_sparsification_samples(
    abs_rel_error: torch.Tensor,
    a1_error: torch.Tensor,
    uncertainty: torch.Tensor,
    finite_mask: torch.Tensor,
    max_samples: Optional[int],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    batch_size = abs_rel_error.shape[0]
    flat_abs_rel = abs_rel_error.detach().flatten(1)
    flat_a1 = a1_error.detach().flatten(1)
    flat_uncertainty = uncertainty.detach().flatten(1)
    flat_mask = finite_mask.flatten(1)
    _, num_pixels = flat_abs_rel.shape

    per_image_max_samples = max_samples
    if max_samples is not None and max_samples > 0:
        per_image_max_samples = max(1, max_samples // max(batch_size, 1))

    if per_image_max_samples is None or per_image_max_samples <= 0:
        sample_width = num_pixels
    else:
        sample_width = min(per_image_max_samples, num_pixels)

    if sample_width < 1:
        return None

    pixel_offsets = torch.arange(num_pixels, device=flat_mask.device).expand(batch_size, -1)
    compact_order = torch.argsort(
        torch.where(flat_mask, pixel_offsets, pixel_offsets + num_pixels),
        dim=1,
    )

    compact_abs_rel = flat_abs_rel.gather(1, compact_order)
    compact_a1 = flat_a1.gather(1, compact_order)
    compact_uncertainty = flat_uncertainty.gather(1, compact_order)

    valid_counts = flat_mask.sum(dim=1)
    sample_offsets = torch.arange(sample_width, device=flat_mask.device).expand(batch_size, -1)
    sample_counts = valid_counts.clamp_max(sample_width)

    if sample_width == 1:
        sample_indices = torch.zeros_like(sample_offsets)
    else:
        uniform_indices = (
            sample_offsets
            * (valid_counts - 1).clamp_min(0).unsqueeze(1)
            // (sample_width - 1)
        )
        sample_indices = torch.where(
            (valid_counts > sample_width).unsqueeze(1),
            uniform_indices,
            sample_offsets,
        )
    sample_indices = sample_indices.clamp_(0, num_pixels - 1)
    sample_mask = sample_offsets < sample_counts.unsqueeze(1)

    return (
        compact_abs_rel.gather(1, sample_indices),
        compact_a1.gather(1, sample_indices),
        compact_uncertainty.gather(1, sample_indices),
        sample_mask,
    )

def _batched_sparsification_ause(
    error: torch.Tensor,
    uncertainty: torch.Tensor,
    sample_mask: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    if error.ndim != 2 or uncertainty.ndim != 2 or sample_mask.ndim != 2:
        raise ValueError("Expected error, uncertainty, and sample_mask to have shape [B, N]")
    if error.shape != uncertainty.shape or error.shape != sample_mask.shape:
        raise ValueError(
            f"Shape mismatch: error {tuple(error.shape)}, "
            f"uncertainty {tuple(uncertainty.shape)}, sample_mask {tuple(sample_mask.shape)}"
        )

    batch_size, num_samples = error.shape
    result = error.new_full((batch_size,), float("nan"), dtype=torch.float64)
    if num_samples < 2 or num_bins <= 0:
        return result

    error = error.float()
    uncertainty = uncertainty.float()
    sample_mask = sample_mask.bool()
    sample_counts = sample_mask.sum(dim=1)
    valid_rows = sample_counts >= 2
    if not valid_rows.any():
        return result

    safe_error = torch.where(sample_mask, error, torch.zeros_like(error))
    neg_inf = torch.full_like(error, float("-inf"))

    uncertainty_order = torch.argsort(
        torch.where(sample_mask, uncertainty, neg_inf),
        dim=1,
        descending=True,
    )
    oracle_order = torch.argsort(
        torch.where(sample_mask, error, neg_inf),
        dim=1,
        descending=True,
    )

    uncertainty_sorted_error = safe_error.gather(1, uncertainty_order)
    oracle_sorted_error = safe_error.gather(1, oracle_order)
    uncertainty_suffix_sum = uncertainty_sorted_error.flip(1).cumsum(dim=1).flip(1)
    oracle_suffix_sum = oracle_sorted_error.flip(1).cumsum(dim=1).flip(1)

    max_points = min(num_bins + 1, num_samples)
    point_counts = torch.minimum(
        sample_counts,
        sample_counts.new_full(sample_counts.shape, max_points),
    )
    point_offsets = torch.arange(max_points, device=error.device, dtype=torch.float64).unsqueeze(0)
    point_denominator = (point_counts - 1).clamp_min(1).to(dtype=torch.float64).unsqueeze(1)
    remove_counts = torch.round(
        point_offsets * (sample_counts - 1).clamp_min(0).to(dtype=torch.float64).unsqueeze(1) / point_denominator
    ).long()
    remove_counts = remove_counts.clamp_(0, num_samples - 1)

    remaining_counts = (sample_counts.unsqueeze(1) - remove_counts).clamp_min(1).to(dtype=torch.float32)
    uncertainty_curve = uncertainty_suffix_sum.gather(1, remove_counts) / remaining_counts
    oracle_curve = oracle_suffix_sum.gather(1, remove_counts) / remaining_counts
    curve_gap = (uncertainty_curve - oracle_curve).double()
    removed_fractions = remove_counts.double() / sample_counts.clamp_min(1).double().unsqueeze(1)

    segment_mask = (
        torch.arange(max_points - 1, device=error.device).unsqueeze(0)
        < (point_counts - 1).unsqueeze(1)
    ) & valid_rows.unsqueeze(1)
    segment_area = (
        0.5
        * (curve_gap[:, :-1] + curve_gap[:, 1:])
        * (removed_fractions[:, 1:] - removed_fractions[:, :-1])
    )
    result = torch.where(
        valid_rows,
        torch.where(segment_mask, segment_area, torch.zeros_like(segment_area)).sum(dim=1),
        result,
    )
    return result


def _batched_sparsification_aurg(
    error: torch.Tensor,
    uncertainty: torch.Tensor,
    sample_mask: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    if error.ndim != 2 or uncertainty.ndim != 2 or sample_mask.ndim != 2:
        raise ValueError("Expected error, uncertainty, and sample_mask to have shape [B, N]")
    if error.shape != uncertainty.shape or error.shape != sample_mask.shape:
        raise ValueError(
            f"Shape mismatch: error {tuple(error.shape)}, "
            f"uncertainty {tuple(uncertainty.shape)}, sample_mask {tuple(sample_mask.shape)}"
        )

    batch_size, num_samples = error.shape
    result = error.new_full((batch_size,), float("nan"), dtype=torch.float64)
    if num_samples < 2 or num_bins <= 0:
        return result

    error = error.float()
    uncertainty = uncertainty.float()
    sample_mask = sample_mask.bool()
    sample_counts = sample_mask.sum(dim=1)
    valid_rows = sample_counts >= 2
    if not valid_rows.any():
        return result

    safe_error = torch.where(sample_mask, error, torch.zeros_like(error))
    neg_inf = torch.full_like(error, float("-inf"))
    uncertainty_order = torch.argsort(
        torch.where(sample_mask, uncertainty, neg_inf),
        dim=1,
        descending=True,
    )
    uncertainty_sorted_error = safe_error.gather(1, uncertainty_order)
    uncertainty_suffix_sum = uncertainty_sorted_error.flip(1).cumsum(dim=1).flip(1)

    max_points = min(num_bins + 1, num_samples)
    point_counts = torch.minimum(
        sample_counts,
        sample_counts.new_full(sample_counts.shape, max_points),
    )
    point_offsets = torch.arange(max_points, device=error.device, dtype=torch.float64).unsqueeze(0)
    point_denominator = (point_counts - 1).clamp_min(1).to(dtype=torch.float64).unsqueeze(1)
    remove_counts = torch.round(
        point_offsets * (sample_counts - 1).clamp_min(0).to(dtype=torch.float64).unsqueeze(1) / point_denominator
    ).long()
    remove_counts = remove_counts.clamp_(0, num_samples - 1)

    remaining_counts = (sample_counts.unsqueeze(1) - remove_counts).clamp_min(1).to(dtype=torch.float32)
    uncertainty_curve = uncertainty_suffix_sum.gather(1, remove_counts) / remaining_counts
    random_curve = safe_error.sum(dim=1, keepdim=True) / sample_counts.clamp_min(1).to(dtype=torch.float32).unsqueeze(1)
    curve_gain = (random_curve - uncertainty_curve).double()
    removed_fractions = remove_counts.double() / sample_counts.clamp_min(1).double().unsqueeze(1)

    segment_mask = (
        torch.arange(max_points - 1, device=error.device).unsqueeze(0)
        < (point_counts - 1).unsqueeze(1)
    ) & valid_rows.unsqueeze(1)
    segment_area = (
        0.5
        * (curve_gain[:, :-1] + curve_gain[:, 1:])
        * (removed_fractions[:, 1:] - removed_fractions[:, :-1])
    )
    result = torch.where(
        valid_rows,
        torch.where(segment_mask, segment_area, torch.zeros_like(segment_area)).sum(dim=1),
        result,
    )
    return result



# @torch.no_grad()
# def compute_image_uncertainty_metric_correlations(
#     image_values: Dict[str, torch.Tensor],
#     prefix: str = "image_mean_uncertainty",
# ) -> Dict[str, float]:
#     """
#     Correlate image-level mean uncertainty with image-level depth metrics.
#     """
#     mean_uncertainty = image_values["mean_uncertainty"].detach().float().cpu()
#     metrics = {
#         "abs_rel": image_values["abs_rel"].detach().float().cpu(),
#         "a1": image_values["a1"].detach().float().cpu(),
#     }

#     result = {
#         prefix: float(mean_uncertainty.mean().item()) if mean_uncertainty.numel() > 0 else float("nan"),
#         f"{prefix}_samples": int(mean_uncertainty.numel()),
#     }

#     for metric_name, metric_values in metrics.items():
#         finite_mask = torch.isfinite(mean_uncertainty) & torch.isfinite(metric_values)
#         x = mean_uncertainty[finite_mask]
#         y = metric_values[finite_mask]

#         if x.numel() < 2:
#             pearson = x.new_tensor(float("nan"))
#             spearman = x.new_tensor(float("nan"))
#         else:
#             pearson = _pearson_corr(x, y)
#             spearman = _spearman_corr(x, y)

#         result[f"{prefix}_{metric_name}_pearson"] = float(pearson.item())
#         result[f"{prefix}_{metric_name}_spearman"] = float(spearman.item())

#     return result


# def gaussian_nll_loss_map(
#     mu: torch.Tensor,
#     log_var: torch.Tensor,
#     target: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Return the per-pixel Gaussian NLL data term without reduction.

#     Shapes:
#         mu:      [B, 1, H, W]
#         log_var: [B, 1, H, W]
#         target:  [B, H, W] or [B, 1, H, W]
#     """
#     if target.ndim == 3:
#         target = target.unsqueeze(1)

#     residual2 = (target - mu) ** 2
#     return 0.5 * (torch.exp(-log_var) * residual2 + log_var)
