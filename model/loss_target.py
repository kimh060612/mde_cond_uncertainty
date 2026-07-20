from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from evaluation_utils.eval_utils import (
    align_relative_prediction_to_depth_space,
)


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(
            f"Expected [B,H,W] or [B,1,H,W], got {tuple(x.shape)}"
        )

    return x.float()


def _coarse_patch_median(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    grid_size: tuple[int, int],
    min_valid_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        pooled_depth: [B, N]
        valid_cells:  [B, N]
    """
    batch_size, _, height, width = depth.shape
    grid_h, grid_w = grid_size

    patch_h = math.ceil(height / grid_h)
    patch_w = math.ceil(width / grid_w)

    pad_h = grid_h * patch_h - height
    pad_w = grid_w * patch_w - width

    depth = F.pad(
        depth,
        (0, pad_w, 0, pad_h),
        value=float("nan"),
    )
    valid_mask = F.pad(
        valid_mask,
        (0, pad_w, 0, pad_h),
        value=False,
    )

    depth_patches = (
        depth.unfold(2, patch_h, patch_h)
        .unfold(3, patch_w, patch_w)
        .flatten(-2)
    )
    mask_patches = (
        valid_mask.unfold(2, patch_h, patch_h)
        .unfold(3, patch_w, patch_w)
        .flatten(-2)
    )

    masked_depth = depth_patches.masked_fill(
        ~mask_patches,
        float("nan"),
    )

    pooled = torch.nanmedian(masked_depth, dim=-1).values
    valid_ratio = mask_patches.float().mean(dim=-1)

    valid_cells = (
        (valid_ratio >= min_valid_ratio)
        & torch.isfinite(pooled)
    )

    pooled = pooled.reshape(batch_size, -1)
    valid_cells = valid_cells.reshape(batch_size, -1)

    return pooled, valid_cells


def _robust_normalize(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Per-image median/IQR normalization.
    """
    normalized = torch.full_like(values, float("nan"))

    for batch_idx in range(values.shape[0]):
        mask = valid_mask[batch_idx]
        valid_values = values[batch_idx, mask]

        if valid_values.numel() < 4:
            continue

        median = valid_values.median()
        q1 = torch.quantile(valid_values, 0.25)
        q3 = torch.quantile(valid_values, 0.75)
        scale = (q3 - q1).clamp_min(eps)

        normalized[batch_idx] = (
            values[batch_idx] - median
        ) / scale

    return normalized


def ordinal_structure_failure(
    source_depth: torch.Tensor,
    canonical_depth: torch.Tensor,
    *,
    grid_size: tuple[int, int] = (12, 16),
    min_valid_ratio: float = 0.5,
    min_canonical_gap: float = 0.25,
    required_retention: float = 0.5,
    temperature: float = 0.1,
    positive_only: bool = True,
    eps: float = 1e-6,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    source_depth = _ensure_bchw(source_depth)
    canonical_depth = _ensure_bchw(canonical_depth)

    if source_depth.shape != canonical_depth.shape:
        raise ValueError(
            "Source and canonical depth shapes must match: "
            f"{tuple(source_depth.shape)} vs "
            f"{tuple(canonical_depth.shape)}"
        )

    source_valid = torch.isfinite(source_depth)
    canonical_valid = torch.isfinite(canonical_depth)

    if positive_only:
        source_valid &= source_depth > 0
        canonical_valid &= canonical_depth > 0

    source_cells, source_cell_valid = _coarse_patch_median(
        source_depth,
        source_valid,
        grid_size,
        min_valid_ratio,
    )
    canonical_cells, canonical_cell_valid = _coarse_patch_median(
        canonical_depth,
        canonical_valid,
        grid_size,
        min_valid_ratio,
    )

    # Both maps must contain a valid value in the same grid cell.
    cell_valid = source_cell_valid & canonical_cell_valid

    source_cells = _robust_normalize(
        source_cells,
        cell_valid,
        eps,
    )
    canonical_cells = _robust_normalize(
        canonical_cells,
        cell_valid,
        eps,
    )

    canonical_gap = (
        canonical_cells.unsqueeze(2)
        - canonical_cells.unsqueeze(1)
    )
    source_gap = (
        source_cells.unsqueeze(2)
        - source_cells.unsqueeze(1)
    )

    num_cells = canonical_cells.shape[1]

    upper_triangle = torch.triu(
        torch.ones(
            num_cells,
            num_cells,
            dtype=torch.bool,
            device=canonical_cells.device,
        ),
        diagonal=1,
    ).unsqueeze(0)

    pair_valid = (
        cell_valid.unsqueeze(2)
        & cell_valid.unsqueeze(1)
        & upper_triangle
        & torch.isfinite(canonical_gap)
        & torch.isfinite(source_gap)
    )

    canonical_strength = canonical_gap.abs()

    # Remove ambiguous canonical near-ties.
    pair_valid &= canonical_strength >= min_canonical_gap

    # Fraction of canonical ordinal separation preserved by source.
    #
    # 1.0: preserved
    # 0.0: collapsed
    # <0 : inverted
    retention = (
        canonical_gap.sign() * source_gap
    ) / canonical_strength.clamp_min(eps)

    # Smooth approximation of:
    # max(required_retention - retention, 0)
    pair_penalty = temperature * F.softplus(
        (required_retention - retention) / temperature
    )

    pair_penalty = torch.where(
        pair_valid,
        pair_penalty,
        torch.zeros_like(pair_penalty),
    )

    pair_count = pair_valid.sum(dim=(1, 2))
    failure_score = pair_penalty.sum(dim=(1, 2)) / (
        pair_count.to(pair_penalty.dtype).clamp_min(1)
    )

    failure_score = torch.where(
        pair_count > 0,
        failure_score,
        torch.full_like(failure_score, float("nan")),
    )

    if not return_details:
        return failure_score

    inversion_count = (
        pair_valid & (retention < 0)
    ).sum(dim=(1, 2))

    collapse_count = (
        pair_valid
        & (retention >= 0)
        & (retention < required_retention)
    ).sum(dim=(1, 2))

    denominator = pair_count.to(pair_penalty.dtype).clamp_min(1)

    details = {
        "hard_inversion_rate": inversion_count / denominator,
        "margin_collapse_rate": collapse_count / denominator,
        "num_valid_pairs": pair_count,
        "mean_retention": torch.where(
            pair_count > 0,
            torch.where(
                pair_valid,
                retention,
                torch.zeros_like(retention),
            ).sum(dim=(1, 2)) / denominator,
            torch.full_like(failure_score, float("nan")),
        ),
    }

    return failure_score, details


def ssi_independent_depth_loss(
    candidate_depth: torch.Tensor,
    canonical_depth: torch.Tensor,
    candidate_gt_depth: torch.Tensor,
    canonical_gt_depth: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    predictions = torch.stack((candidate_depth, canonical_depth), dim=1)
    targets = torch.stack((candidate_gt_depth, canonical_gt_depth), dim=1)
    valid = (
        torch.isfinite(predictions)
        & torch.isfinite(targets)
        & (predictions > 0)
        & (targets > 0)
    )

    valid_float = valid.to(predictions.dtype)
    counts = valid_float.flatten(2).sum(dim=2)
    safe_counts = counts.clamp_min(1.0)
    prediction_values = torch.where(valid, predictions, torch.zeros_like(predictions))
    target_values = torch.where(valid, targets, torch.zeros_like(targets))

    sum_prediction = prediction_values.flatten(2).sum(dim=2)
    sum_target = target_values.flatten(2).sum(dim=2)
    sum_prediction_squared = prediction_values.square().flatten(2).sum(dim=2)
    sum_prediction_target = (prediction_values * target_values).flatten(2).sum(dim=2)

    denominator = safe_counts * sum_prediction_squared - sum_prediction.square()
    stable = (counts > 1) & (denominator.abs() > eps)
    scale = torch.where(
        stable,
        (safe_counts * sum_prediction_target - sum_prediction * sum_target)
        / denominator.clamp_min(eps),
        torch.zeros_like(counts),
    )
    shift = torch.where(
        counts > 0,
        (sum_target - scale * sum_prediction) / safe_counts,
        torch.zeros_like(counts),
    )

    aligned = (
        scale[:, :, None, None, None] * predictions
        + shift[:, :, None, None, None]
    )
    comparison_valid = valid[:, 0] & valid[:, 1]
    valid_counts = comparison_valid.flatten(1).sum(dim=1)
    difference = torch.where(
        comparison_valid,
        (aligned[:, 0] - aligned[:, 1]).abs(),
        torch.zeros_like(aligned[:, 0]),
    )
    loss = difference.flatten(1).sum(dim=1) / valid_counts.clamp_min(1)
    return torch.where(
        valid_counts > 0,
        loss,
        torch.full_like(loss, float("nan")),
    )


def ssi_independent_meter_space_depth_loss(
    candidate_depth: torch.Tensor,
    canonical_depth: torch.Tensor,
    candidate_gt_depth: torch.Tensor,
    canonical_gt_depth: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    candidate_depth = _ensure_bchw(candidate_depth)
    canonical_depth = _ensure_bchw(canonical_depth)
    candidate_gt_depth = _ensure_bchw(candidate_gt_depth)
    canonical_gt_depth = _ensure_bchw(canonical_gt_depth)

    candidate_valid = (
        torch.isfinite(candidate_depth)
        & torch.isfinite(candidate_gt_depth)
        & (candidate_depth > 0)
        & (candidate_gt_depth > 0)
    )
    canonical_valid = (
        torch.isfinite(canonical_depth)
        & torch.isfinite(canonical_gt_depth)
        & (canonical_depth > 0)
        & (canonical_gt_depth > 0)
    )

    candidate_meter_depth = align_relative_prediction_to_depth_space(
        candidate_depth,
        candidate_gt_depth,
        candidate_valid,
        align_mode="scale_shift",
        eps=eps,
    )["depth"]
    canonical_meter_depth = align_relative_prediction_to_depth_space(
        canonical_depth,
        canonical_gt_depth,
        canonical_valid,
        align_mode="scale_shift",
        eps=eps,
    )["depth"]

    comparison_valid = (
        candidate_valid
        & canonical_valid
        & torch.isfinite(candidate_meter_depth)
        & torch.isfinite(canonical_meter_depth)
        & (candidate_meter_depth > 0)
        & (canonical_meter_depth > 0)
    )
    valid_counts = comparison_valid.flatten(1).sum(dim=1)
    difference = torch.where(
        comparison_valid,
        (candidate_meter_depth - canonical_meter_depth).abs() / canonical_meter_depth.clamp_min(eps),
        torch.zeros_like(candidate_meter_depth),
    )
    loss = difference.flatten(1).sum(dim=1) / valid_counts.clamp_min(1)
    return torch.where(
        valid_counts > 0,
        loss,
        torch.full_like(loss, float("nan")),
    )
