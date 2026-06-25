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
from typing import Optional, Tuple
from evaluation_utils.eval_utils import ensure_bchw
import torch

def prepare_bchw_mask(
    valid_mask: torch.Tensor, 
    reference: torch.Tensor
) -> torch.Tensor:
    valid_mask = ensure_bchw(valid_mask)
    mask = valid_mask.bool()
    if mask.shape != reference.shape:
        mask = mask.expand_as(reference)
    return mask


def deterministic_subsample(
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

def calculate_l1_depth_lossmap(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mu = ensure_bchw(mu)
    target = ensure_bchw(target)
    valid_mask = prepare_bchw_mask(valid_mask, target)
    l1_loss_map = torch.abs(mu - target)
    return l1_loss_map

def prepare_masked_vectors(
    x: torch.Tensor,
    y: torch.Tensor,
    valid_mask: torch.Tensor,
    max_samples: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = prepare_bchw_mask(valid_mask, x)

    finite_mask = mask & torch.isfinite(x) & torch.isfinite(y)
    x = x[finite_mask].detach().flatten()
    y = y[finite_mask].detach().flatten()

    x, y = deterministic_subsample(x, y, max_samples=max_samples)

    return x.float(), y.float()


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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


def ordinal_ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float64)
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2:
        return x.new_tensor(float("nan"))

    return pearson_corr(ordinal_ranks(x), ordinal_ranks(y))

def batched_pearson_corr(
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


def batched_ordinal_ranks(x: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
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


def batched_spearman_corr(
    x: torch.Tensor,
    y: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    return batched_pearson_corr(
        batched_ordinal_ranks(x, sample_mask),
        batched_ordinal_ranks(y, sample_mask),
        sample_mask,
    )

def prepare_sparsification_samples(
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

def batched_sparsification_ause(
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


def batched_sparsification_aurg(
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
