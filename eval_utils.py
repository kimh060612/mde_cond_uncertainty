import torch
import math

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
