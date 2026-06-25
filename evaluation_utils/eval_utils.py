from typing import Dict, Tuple, Optional
import math
import torch


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
    inv_depth_min: float = 1e-6,
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
