import math

import torch


def _as_bchw(x):
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected [B, H, W] or [B, 1, H, W], got {tuple(x.shape)}")


def _nan_depth_metrics():
    return {
        "abs_rel": float("nan"),
        "sq_rel": float("nan"),
        "rmse": float("nan"),
        "rmse_log": float("nan"),
        "a1": float("nan"),
        "a2": float("nan"),
        "a3": float("nan"),
    }


@torch.no_grad()
def __compute_errors_numpy(
    gt,
    pred,
    min_depth=1e-3,
    max_depth=80.0,
    align_mode=None,   # None, "median", "scale_shift"
    eps=1e-8,
):
    gt = torch.as_tensor(gt, dtype=torch.float64)
    pred = torch.as_tensor(pred, dtype=torch.float64)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape}, pred {pred.shape}")

    # 1) valid mask
    valid = torch.isfinite(gt) & torch.isfinite(pred)
    valid &= (gt > 0) & (gt < float('inf')) & (pred < float('inf'))
    valid &= (pred > 0)
    if valid.sum() == 0:
        return _nan_depth_metrics()

    gt_valid = gt[valid]
    gt_inv_valid = 1.0 / (gt_valid + eps)
    pred_valid = pred[valid]

    # 2) optional alignment for relative-depth prediction
    if align_mode is not None:
        if align_mode == "median":
            scale = torch.median(gt_inv_valid) / (torch.median(pred_valid) + eps)
            pred_valid = pred_valid * scale

        elif align_mode == "scale_shift":
            # solve: gt_inv ≈ s * pred + t
            A = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)  # [N, 2]
            x = torch.linalg.lstsq(A, gt_inv_valid.unsqueeze(1)).solution
            s, t = x[:2, 0]
            pred_valid = s * pred_valid + t
            pred_valid = torch.clamp_min(pred_valid, 1e-6)

        else:
            raise ValueError(f"Unknown align_mode: {align_mode}")

    # 3) clamp after alignment
    pred_valid = 1. / (pred_valid + 1e-8)
    pred_valid = torch.clamp(pred_valid, min_depth, max_depth)
    gt_valid = torch.clamp(gt_valid, min_depth, max_depth)

    # 4) metrics
    thresh = torch.maximum(gt_valid / (pred_valid + eps), pred_valid / (gt_valid + eps))
    a1 = torch.mean((thresh < 1.25).float())
    a2 = torch.mean((thresh < 1.25 ** 2).float())
    a3 = torch.mean((thresh < 1.25 ** 3).float())

    rmse = torch.sqrt(torch.mean((gt_valid - pred_valid) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(gt_valid + eps) - torch.log(pred_valid + eps)) ** 2))

    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / (gt_valid + eps))
    sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / (gt_valid + eps))

    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "a1": float(a1),
        "a2": float(a2),
        "a3": float(a3),
    }

@torch.no_grad()
def compute_metrics(mu, target, valid_mask):
    target = target.unsqueeze(1)
    valid_mask = valid_mask.unsqueeze(1).bool()

    pred = mu[valid_mask]
    gt = target[valid_mask]

    pred = pred.clamp_min(1e-3)
    gt = gt.clamp_min(1e-3)

    abs_rel = torch.mean(torch.abs(pred - gt) / gt)
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))

    ratio = torch.maximum(pred / gt, gt / pred)
    a1 = torch.mean((ratio < 1.25).float())

    return {
        "abs_rel": abs_rel.item(),
        "rmse": rmse.item(),
        "a1": a1.item(),
    }


@torch.no_grad()
def compute_relative_depth_metrics(
    mu,
    target,
    valid_mask,
    min_depth=1e-3,
    max_depth=80.0,
    align_mode="scale_shift",
):
    pred = _as_bchw(mu.detach())
    gt = _as_bchw(target.detach())
    valid_mask = _as_bchw(valid_mask).bool()

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: target {gt.shape}, pred {pred.shape}")
    if valid_mask.shape != pred.shape:
        valid_mask = valid_mask.expand_as(pred)

    metric_sums = {}
    metric_counts = {}

    # Relative-depth scale is image-dependent, so align each image separately.
    for batch_idx in range(pred.shape[0]):
        mask = valid_mask[batch_idx]
        if mask.sum() == 0:
            continue

        image_metrics = __compute_errors_numpy(
            gt=gt[batch_idx][mask].cpu(),
            pred=pred[batch_idx][mask].cpu(),
            min_depth=min_depth,
            max_depth=max_depth,
            align_mode=align_mode,
        )
        _accumulate_finite_metrics(metric_sums, metric_counts, image_metrics)

    if not metric_sums:
        return _nan_depth_metrics()

    metrics = _nan_depth_metrics()
    metrics.update(_mean_finite_metrics(metric_sums, metric_counts))
    return metrics


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
