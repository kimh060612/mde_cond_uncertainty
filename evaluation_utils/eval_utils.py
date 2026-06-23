from typing import Dict, Tuple, Optional
import math
import torch


def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected [B, H, W] or [B, 1, H, W], got {tuple(x.shape)}")

def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_counts = mask.flatten(1).sum(dim=1)
    sums = torch.where(mask, values, torch.zeros_like(values)).flatten(1).sum(dim=1)
    means = sums / valid_counts.clamp_min(1).to(dtype=values.dtype)
    return torch.where(valid_counts > 0, means, torch.full_like(means, float("nan")))

def _masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

def _metric_dict(
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
        "abs_rel": _masked_mean(torch.abs(diff) / (gt_depth + eps), mask),
        "rmse": torch.sqrt(_masked_mean(sq_error, mask)),
        "a1": _masked_mean((thresh < 1.25).to(dtype=calc_dtype), mask),
        "a2": _masked_mean((thresh < 1.25 ** 2).to(dtype=calc_dtype), mask),
        "a3": _masked_mean((thresh < 1.25 ** 3).to(dtype=calc_dtype), mask),
    }

@torch.no_grad()
def compute_comprehensive_depth_metrics(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_model_type: str,
    min_depth=1e-3,
    max_depth=80.0,
    align_mode="scale_shift",
) -> Dict[str, torch.Tensor]:
    """
    Calculating depth metrics in parallel manner with CUDA GPU and PyTorch. 
    This function computes both absolute depth metrics and relative depth metrics for a batch of images.
    Returning Dictionary of tensors of each metrics. (Abs_rel, RMSE, A1, A2, A3)
    
        Input: 
            - mu: Predicted depth map tensor of shape [B, 1, H, W]
            - target: Ground truth depth map tensor of shape [B, 1, H, W]
            - valid_mask: Boolean tensor indicating valid pixels of shape [B, 1, H, W]
            - depth_model_type: str, either "metric" or "relative"
            - min_depth: Minimum depth value for clamping
            - max_depth: Maximum depth value for clamping
            - align_mode: Alignment mode for relative depth metrics, either "median" or "scale_shift"       
    
        Output:
            - metrics: Dictionary containing computed depth metrics 
            Key: "abs_rel", "rmse", "sq_rel", "a1", "a2", "a3"
            Values: Tensor of shape [B] containing the metric for each image in the batch
    """
    pred = _as_bchw(mu.detach()) # [B, 1, H, W]
    gt = _as_bchw(target.detach()) # [B, 1, H, W]
    valid_mask = _as_bchw(valid_mask).bool()

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: target {gt.shape}, pred {pred.shape}")
    if valid_mask.shape != pred.shape:
        valid_mask = valid_mask.expand_as(pred)

    calc_dtype = torch.float64 if pred.dtype == torch.float64 or gt.dtype == torch.float64 else torch.float32
    pred = pred.to(dtype=calc_dtype)
    gt = gt.to(dtype=calc_dtype)
    eps = 1e-8

    depth_model_type = depth_model_type.lower()

    if depth_model_type == "metric":
        metric_mask = valid_mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0)
        return _metric_dict(
            pred, 
            gt, 
            metric_mask, 
            min_depth=min_depth, 
            max_depth=max_depth, 
            eps=eps, 
            calc_dtype=calc_dtype
        )
    elif depth_model_type == "relative":
        relative_mask = valid_mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0) & (pred > 0)
        gt_inv = 1.0 / (gt + eps)

        if align_mode == "median":
            pred_median = _masked_median(pred, relative_mask)
            gt_inv_median = _masked_median(gt_inv, relative_mask)
            scale = gt_inv_median / (pred_median + eps)
            pred_aligned = pred * scale.view(-1, 1, 1, 1)
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
            pred_aligned = pred * scale.view(-1, 1, 1, 1) + shift.view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown align_mode: {align_mode}")

        pred_depth = 1.0 / pred_aligned.clamp_min(1e-6)
        return _metric_dict(
            pred_depth, 
            gt, 
            relative_mask, 
            min_depth=min_depth, 
            max_depth=max_depth, 
            eps=eps, 
            calc_dtype=calc_dtype
        )
    else:
        raise NotImplementedError(f"Unknown depth_model_type: {depth_model_type}")


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


# def _nan_depth_metrics():
#     return {
#         "abs_rel": float("nan"),
#         "sq_rel": float("nan"),
#         "rmse": float("nan"),
#         "rmse_log": float("nan"),
#         "a1": float("nan"),
#         "a2": float("nan"),
#         "a3": float("nan"),
#     }


# @torch.no_grad()
# def __compute_errors_numpy(
#     gt,
#     pred,
#     min_depth=1e-3,
#     max_depth=80.0,
#     align_mode=None,   # None, "median", "scale_shift"
#     eps=1e-8,
# ):
#     gt = torch.as_tensor(gt, dtype=torch.float64).to(pred.device)
#     pred = torch.as_tensor(pred, dtype=torch.float64).to(gt.device)
#     if gt.shape != pred.shape:
#         raise ValueError(f"Shape mismatch: gt {gt.shape}, pred {pred.shape}")

#     # 1) valid mask
#     valid = torch.isfinite(gt) & torch.isfinite(pred)
#     valid &= (gt > 0) & (gt < float('inf')) & (pred < float('inf'))
#     valid &= (pred > 0)
#     if valid.sum() == 0:
#         return _nan_depth_metrics()

#     gt_valid = gt[valid]
#     gt_inv_valid = 1.0 / (gt_valid + eps)
#     pred_valid = pred[valid]

#     # 2) optional alignment for relative-depth prediction
#     if align_mode is not None:
#         if align_mode == "median":
#             scale = torch.median(gt_inv_valid) / (torch.median(pred_valid) + eps)
#             pred_valid = pred_valid * scale

#         elif align_mode == "scale_shift":
#             # solve: gt_inv ≈ s * pred + t
#             A = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)  # [N, 2]
#             x = torch.linalg.lstsq(A, gt_inv_valid.unsqueeze(1)).solution
#             s, t = x[:2, 0]
#             pred_valid = s * pred_valid + t
#             pred_valid = torch.clamp_min(pred_valid, 1e-6)

#         else:
#             raise ValueError(f"Unknown align_mode: {align_mode}")

#     # 3) clamp after alignment
#     pred_valid = 1. / (pred_valid + 1e-8)
#     pred_valid = torch.clamp(pred_valid, min_depth, max_depth)
#     gt_valid = torch.clamp(gt_valid, min_depth, max_depth)

#     # 4) metrics
#     thresh = torch.maximum(gt_valid / (pred_valid + eps), pred_valid / (gt_valid + eps))
#     a1 = torch.mean((thresh < 1.25).float())
#     a2 = torch.mean((thresh < 1.25 ** 2).float())
#     a3 = torch.mean((thresh < 1.25 ** 3).float())

#     rmse = torch.sqrt(torch.mean((gt_valid - pred_valid) ** 2))
#     rmse_log = torch.sqrt(torch.mean((torch.log(gt_valid + eps) - torch.log(pred_valid + eps)) ** 2))

#     abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / (gt_valid + eps))
#     sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / (gt_valid + eps))

#     return {
#         "abs_rel": float(abs_rel),
#         "sq_rel": float(sq_rel),
#         "rmse": float(rmse),
#         "rmse_log": float(rmse_log),
#         "a1": float(a1),
#         "a2": float(a2),
#         "a3": float(a3),
#     }

# @torch.no_grad()
# def compute_metrics(mu, target, valid_mask):
#     target = target.unsqueeze(1)
#     valid_mask = valid_mask.unsqueeze(1).bool()

#     pred = mu[valid_mask]
#     gt = target[valid_mask]

#     pred = pred.clamp_min(1e-3)
#     gt = gt.clamp_min(1e-3)

#     abs_rel = torch.mean(torch.abs(pred - gt) / gt)
#     rmse = torch.sqrt(torch.mean((pred - gt) ** 2))

#     ratio = torch.maximum(pred / gt, gt / pred)
#     a1 = torch.mean((ratio < 1.25).float())

#     return {
#         "abs_rel": abs_rel.item(),
#         "rmse": rmse.item(),
#         "a1": a1.item(),
#     }



# @torch.no_grad()
# def compute_relative_depth_metrics(
#     mu,
#     target,
#     valid_mask,
#     min_depth=1e-3,
#     max_depth=80.0,
#     align_mode="scale_shift",
# ):
#     pred = _as_bchw(mu.detach())
#     gt = _as_bchw(target.detach())
#     valid_mask = _as_bchw(valid_mask).bool()

#     if pred.shape != gt.shape:
#         raise ValueError(f"Shape mismatch: target {gt.shape}, pred {pred.shape}")
#     if valid_mask.shape != pred.shape:
#         valid_mask = valid_mask.expand_as(pred)

#     metric_sums = {}
#     metric_counts = {}

#     # Relative-depth scale is image-dependent, so align each image separately.
#     for batch_idx in range(pred.shape[0]):
#         mask = valid_mask[batch_idx]
#         if mask.sum() == 0:
#             continue

#         image_metrics = __compute_errors_numpy(
#             gt=gt[batch_idx][mask],
#             pred=pred[batch_idx][mask],
#             min_depth=min_depth,
#             max_depth=max_depth,
#             align_mode=align_mode,
#         )
#         _accumulate_finite_metrics(metric_sums, metric_counts, image_metrics)

#     if not metric_sums:
#         return _nan_depth_metrics()

#     metrics = _nan_depth_metrics()
#     metrics.update(_mean_finite_metrics(metric_sums, metric_counts))
#     return metrics
