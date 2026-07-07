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
from evaluation_utils.eval_utils import metric_dict, depth_error_maps, ensure_bchw
from evaluation_utils.correlation_utils import *
import torch

### Depth Estimation Metrics
@torch.no_grad()
def compute_comprehensive_depth_metrics(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    min_depth=1e-3,
    max_depth=80.0,
) -> Dict[str, torch.Tensor]:
    """
    Calculating depth metrics in parallel manner with CUDA GPU and PyTorch. 
    This function computes both absolute depth metrics and relative depth metrics for a batch of images.
    Returning Dictionary of tensors of each metrics. (Abs_rel, RMSE, A1, A2, A3)
    
        Input: 
            - mu: Predicted depth map tensor of shape [B, 1, H, W]
            - target: Ground truth depth map tensor of shape [B, 1, H, W]
            - valid_mask: Boolean tensor indicating valid pixels of shape [B, 1, H, W]
            - min_depth: Minimum depth value for clamping
            - max_depth: Maximum depth value for clamping 
    
        Output:
            - metrics: Dictionary containing computed depth metrics 
            Key: "abs_rel", "rmse", "sq_rel", "a1", "a2", "a3"
            Values: Tensor of shape [B] containing the metric for each image in the batch
    """
    pred = ensure_bchw(mu.detach())
    gt = ensure_bchw(target.detach())
    valid_mask = ensure_bchw(valid_mask).bool()
    calc_dtype = torch.float64 if mu.dtype == torch.float64 or target.dtype == torch.float64 else torch.float32
    pred = pred.to(dtype=calc_dtype)
    gt = gt.to(dtype=calc_dtype)
    if pred.shape != gt.shape:
        raise ValueError(...)
    if valid_mask.shape != pred.shape:
        valid_mask = valid_mask.expand_as(pred)
    
    eps = 1e-8
    valid_mask = valid_mask.bool()
    metric_mask = valid_mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0)
    return metric_dict(
        pred, 
        gt, 
        metric_mask, 
        min_depth=min_depth, 
        max_depth=max_depth, 
        eps=eps, 
        calc_dtype=calc_dtype
    )

### AUSE, AURG metric
@torch.no_grad()
def compute_sparsification_ause_metrics(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    max_samples: Optional[int] = 100_000,
    num_bins: int = 100,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute MDE-style AUSE metrics by removing high-uncertainty pixels first.

    ``ause_abs_rel`` uses per-pixel AbsRel error. ``ause_a1`` uses the a1
    failure indicator, so lower AUSE is better for both metrics.
    """
    
    abs_rel_error, a1_error = depth_error_maps(
        mu,
        target,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    uncertainty = ensure_bchw(uncertainty)
    if uncertainty.shape != abs_rel_error.shape:
        uncertainty = uncertainty.expand_as(abs_rel_error)

    mask = prepare_bchw_mask(valid_mask, abs_rel_error)
    finite_mask = (
        mask
        & torch.isfinite(abs_rel_error)
        & torch.isfinite(a1_error)
        & torch.isfinite(uncertainty)
    )

    samples = prepare_sparsification_samples(
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

    ause_abs_rel = batched_sparsification_ause(
        abs_rel_samples,
        uncertainty_samples,
        sample_mask,
        num_bins=num_bins,
    )
    ause_a1 = batched_sparsification_ause(
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
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute MDE-style AURG metrics by removing high-uncertainty pixels first.

    ``aurg_abs_rel`` uses per-pixel AbsRel error. ``aurg_a1`` uses the a1
    failure indicator, so higher AURG is better for both metrics.
    """
    abs_rel_error, a1_error = depth_error_maps(
        mu,
        target,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    uncertainty = ensure_bchw(uncertainty)
    if uncertainty.shape != abs_rel_error.shape:
        uncertainty = uncertainty.expand_as(abs_rel_error)

    mask = prepare_bchw_mask(valid_mask, abs_rel_error)
    finite_mask = (
        mask
        & torch.isfinite(abs_rel_error)
        & torch.isfinite(a1_error)
        & torch.isfinite(uncertainty)
    )

    samples = prepare_sparsification_samples(
        abs_rel_error,
        a1_error,
        uncertainty,
        finite_mask,
        max_samples=max_samples
    )
    if samples is None:
        empty = abs_rel_error.new_full((abs_rel_error.shape[0],), float("nan"), dtype=torch.float64)
        return {
            "aurg_abs_rel": empty,
            "aurg_a1": empty,
            "aurg_samples": 0,
        }
    abs_rel_samples, a1_samples, uncertainty_samples, sample_mask = samples

    aurg_abs_rel = batched_sparsification_aurg(
        abs_rel_samples,
        uncertainty_samples,
        sample_mask,
        num_bins=num_bins,
    )
    aurg_a1 = batched_sparsification_aurg(
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
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> Dict[str, torch.Tensor]:
    gt = ensure_bchw(gt)
    valid_mask = prepare_bchw_mask(valid_mask, gt)
    l1_loss_map = calculate_l1_depth_lossmap(
        mu,
        gt,
        valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    uncertainty = ensure_bchw(uncertainty)
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
def compute_vector_masked_correlations(
    x: torch.Tensor,
    y: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    max_samples: Optional[int] = 100_000,
    prefix: str = "correlation",
) -> Dict[str, float]:
    """
    Compute Pearson/Spearman correlations for image-level 1D vectors.

    Intended for cases like:
        abs_rel: [N]
        a1: [N]
        uncertainty_mean: [N]

    Unlike compute_masked_correlations(), this does not expect BCHW tensors.
    """
    x = x.detach().flatten()
    y = y.detach().flatten()

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x {tuple(x.shape)} != y {tuple(y.shape)}")

    finite_mask = torch.isfinite(x) & torch.isfinite(y)

    if valid_mask is not None:
        valid_mask = valid_mask.detach().flatten().bool()
        if valid_mask.shape != x.shape:
            raise ValueError(
                f"Shape mismatch: valid_mask {tuple(valid_mask.shape)} != x {tuple(x.shape)}"
            )
        finite_mask = finite_mask & valid_mask

    x = x[finite_mask].float()
    y = y[finite_mask].float()

    x, y = deterministic_subsample(x, y, max_samples=max_samples)

    if x.numel() < 2:
        return {
            f"{prefix}_pearson": float("nan"),
            f"{prefix}_spearman": float("nan"),
        }
    pearson = pearson_corr(x, y)
    spearman = spearman_corr(x, y)

    return {
        f"{prefix}_pearson": float(pearson.item()),
        f"{prefix}_spearman": float(spearman.item()),
    }

@torch.no_grad()
def compute_loss_uncertainty_correlations(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    max_samples: Optional[int] = 100_000,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute correlation between per-pixel L1 depth loss and uncertainty.

    Args:
        uncertainty:
            Per-pixel uncertainty map.
    """
    loss_map = calculate_l1_depth_lossmap(
        mu,
        target,
        valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    uncertainty = ensure_bchw(uncertainty)
    if uncertainty.shape != loss_map.shape:
        uncertainty = uncertainty.expand_as(loss_map)

    mask = prepare_bchw_mask(valid_mask, loss_map)
    finite_mask = mask & torch.isfinite(loss_map) & torch.isfinite(uncertainty)
    samples = prepare_sparsification_samples(
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
        "loss_uncertainty_pearson": batched_pearson_corr(
            loss_samples,
            uncertainty_samples,
            sample_mask,
        ),
        "loss_uncertainty_spearman": batched_spearman_corr(
            loss_samples,
            uncertainty_samples,
            sample_mask,
        ),
    }


@torch.no_grad()
def compute_camera_induced_degradation_values(
    candidate_metrics: Dict[str, torch.Tensor],
    canonical_metrics: Dict[str, torch.Tensor],
    camera_bias: torch.Tensor,
    variance: torch.Tensor,
    valid_mask: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    bias = ensure_bchw(camera_bias.detach())
    variance = ensure_bchw(variance.detach())
    mask = prepare_bchw_mask(valid_mask, bias)

    counts = mask.flatten(1).sum(dim=1)
    safe_counts = counts.clamp_min(1).to(dtype=bias.dtype)
    B2 = torch.where(mask, bias.square(), torch.zeros_like(bias)).flatten(1).sum(dim=1) / safe_counts
    V = torch.where(mask, variance, torch.zeros_like(variance)).flatten(1).sum(dim=1) / safe_counts
    B2 = torch.where(counts > 0, B2, B2.new_full(B2.shape, float("nan")))
    V = torch.where(counts > 0, V, V.new_full(V.shape, float("nan")))
    R = B2 + V

    abs_rel_degradation = candidate_metrics["abs_rel"] - canonical_metrics["abs_rel"]
    delta1_degradation = canonical_metrics["a1"] - candidate_metrics["a1"]
    delta1_error_degradation = (1.0 - candidate_metrics["a1"]) - (1.0 - canonical_metrics["a1"])

    return {
        "B2": B2.detach(),
        "V": V.detach(),
        "R": R.detach(),
        "sqrt_R": torch.sqrt(R.clamp_min(eps)).detach(),
        "log_R": torch.log(R.clamp_min(eps)).detach(),
        "abs_rel_degradation": abs_rel_degradation.detach(),
        "delta1_degradation": delta1_degradation.detach(),
        "delta1_error_degradation": delta1_error_degradation.detach(),
        "group_id": group_ids.detach().flatten(),
    }


@torch.no_grad()
def summarize_camera_induced_degradation_correlations(
    values: Dict[str, torch.Tensor],
    max_samples: Optional[int] = 100_000,
    relative_tie_margin: float = 0.05,
    eps: float = 1e-8,
) -> Dict[str, float]:
    logged_metric_names = (
        "selection_mean_regret_R_vs_abs_rel_degradation",
        "pairwise_accuracy_R_vs_abs_rel_degradation",
        "groupwise_mean_spearman_R_vs_abs_rel_degradation",
        "pearson_R_vs_abs_rel_degradation",
        "spearman_R_vs_abs_rel_degradation",
    )
    score_names = ("R", "sqrt_R", "log_R", "B2", "V")
    target_names = ("abs_rel_degradation", "delta1_degradation", "delta1_error_degradation")
    group_score_names = ("R", "B2", "V")
    grouped_target_names = ("abs_rel_degradation", "delta1_degradation", "delta1_error_degradation")

    if not values or "group_id" not in values:
        return {}

    prepared = {key: tensor.detach().flatten().float().cpu() for key, tensor in values.items()}
    group_ids = prepared["group_id"]
    num_total_samples = int(group_ids.numel())
    primary_valid = torch.isfinite(group_ids)
    for key in ("R", "abs_rel_degradation", "delta1_degradation", "delta1_error_degradation"):
        primary_valid = primary_valid & torch.isfinite(prepared[key])

    metrics: Dict[str, float] = {
        "num_total_samples": float(num_total_samples),
        "num_valid_samples": float(primary_valid.sum().item()),
        "num_invalid_samples": float(num_total_samples - int(primary_valid.sum().item())),
    }

    if primary_valid.any():
        metrics["negative_abs_rel_degradation_ratio"] = float((prepared["abs_rel_degradation"][primary_valid] < 0).float().mean().item())
        metrics["negative_delta_degradation_ratio"] = float((prepared["delta1_degradation"][primary_valid] < 0).float().mean().item())
    else:
        metrics["negative_abs_rel_degradation_ratio"] = float("nan")
        metrics["negative_delta_degradation_ratio"] = float("nan")

    for score_name in score_names:
        for target_name in target_names:
            score = prepared[score_name]
            target = prepared[target_name]
            valid = torch.isfinite(score) & torch.isfinite(target) & torch.isfinite(group_ids)
            key = f"{score_name}_vs_{target_name}"
            if valid.sum().item() < 3:
                metrics[f"pearson_{key}"] = float("nan")
                metrics[f"spearman_{key}"] = float("nan")
                metrics[f"num_samples_{key}"] = float(valid.sum().item())
                continue

            corr = compute_vector_masked_correlations(score, target, valid_mask=valid, max_samples=max_samples, prefix=key)
            metrics[f"pearson_{key}"] = corr[f"{key}_pearson"]
            metrics[f"spearman_{key}"] = corr[f"{key}_spearman"]
            metrics[f"num_samples_{key}"] = float(valid.sum().item())

    for target_name in target_names:
        r_key = f"R_vs_{target_name}"
        sqrt_key = f"sqrt_R_vs_{target_name}"
        r_spearman = metrics.get(f"spearman_{r_key}", float("nan"))
        sqrt_spearman = metrics.get(f"spearman_{sqrt_key}", float("nan"))
        if torch.isfinite(torch.tensor(r_spearman)) and torch.isfinite(torch.tensor(sqrt_spearman)):
            metrics[f"spearman_R_sqrt_R_diff_{target_name}"] = abs(r_spearman - sqrt_spearman)

    for score_name in group_score_names:
        for target_name in grouped_target_names:
            score = prepared[score_name]
            target = prepared[target_name]
            valid = torch.isfinite(score) & torch.isfinite(target) & torch.isfinite(group_ids)
            key = f"{score_name}_vs_{target_name}"

            centered_scores = []
            centered_targets = []
            group_spearman_values = []
            num_centered_groups = 0
            pair_total = 0
            pair_valid = 0
            pair_tied = 0
            pair_correct = 0
            regrets = []
            normalized_regrets = []
            top1_hits = []
            zero_regret_hits = []

            for group_id in torch.unique(group_ids[valid]):
                group_mask = valid & (group_ids == group_id)
                group_score = score[group_mask]
                group_target = target[group_mask]
                group_count = int(group_score.numel())
                if group_count < 2:
                    continue

                centered_scores.append(group_score - group_score.mean())
                centered_targets.append(group_target - group_target.mean())
                num_centered_groups += 1

                if group_count >= 3 and (group_score.max() - group_score.min()).abs() > eps and (group_target.max() - group_target.min()).abs() > eps:
                    group_spearman = spearman_corr(group_score, group_target)
                    if torch.isfinite(group_spearman):
                        group_spearman_values.append(group_spearman)

                for left in range(group_count - 1):
                    for right in range(left + 1, group_count):
                        pair_total += 1
                        target_gap = torch.abs(group_target[left] - group_target[right]) / (torch.abs(group_target[left]) + torch.abs(group_target[right]) + eps)
                        if target_gap < relative_tie_margin:
                            pair_tied += 1
                            continue
                        pair_valid += 1
                        if torch.sign(group_score[left] - group_score[right]).item() == torch.sign(group_target[left] - group_target[right]).item():
                            pair_correct += 1

                predicted_index = torch.argmin(group_score)
                best_target = group_target.min()
                regret = group_target[predicted_index] - best_target
                target_range = group_target.max() - best_target
                regrets.append(regret)
                normalized_regrets.append(regret / (target_range + eps))
                top1_hits.append((group_target[predicted_index] <= best_target + eps).float())
                zero_regret_hits.append((regret.abs() <= eps).float())

            if centered_scores:
                centered_score = torch.cat(centered_scores, dim=0)
                centered_target = torch.cat(centered_targets, dim=0)
                if centered_score.numel() >= 3:
                    metrics[f"group_centered_pearson_{key}"] = float(pearson_corr(centered_score, centered_target).item())
                else:
                    metrics[f"group_centered_pearson_{key}"] = float("nan")
                metrics[f"group_centered_num_samples_{key}"] = float(centered_score.numel())
                metrics[f"group_centered_num_groups_{key}"] = float(num_centered_groups)
            else:
                metrics[f"group_centered_pearson_{key}"] = float("nan")
                metrics[f"group_centered_num_samples_{key}"] = 0.0
                metrics[f"group_centered_num_groups_{key}"] = 0.0

            if group_spearman_values:
                group_spearman_tensor = torch.stack(group_spearman_values).float()
                metrics[f"groupwise_mean_spearman_{key}"] = float(group_spearman_tensor.mean().item())
                metrics[f"groupwise_median_spearman_{key}"] = float(group_spearman_tensor.median().item())
                metrics[f"groupwise_std_spearman_{key}"] = float(group_spearman_tensor.std(unbiased=False).item())
                metrics[f"groupwise_num_valid_groups_{key}"] = float(group_spearman_tensor.numel())
            else:
                metrics[f"groupwise_mean_spearman_{key}"] = float("nan")
                metrics[f"groupwise_median_spearman_{key}"] = float("nan")
                metrics[f"groupwise_std_spearman_{key}"] = float("nan")
                metrics[f"groupwise_num_valid_groups_{key}"] = 0.0

            metrics[f"pairwise_num_total_pairs_{key}"] = float(pair_total)
            metrics[f"pairwise_num_valid_pairs_{key}"] = float(pair_valid)
            metrics[f"pairwise_num_tied_pairs_{key}"] = float(pair_tied)
            metrics[f"pairwise_num_correct_pairs_{key}"] = float(pair_correct)
            metrics[f"pairwise_accuracy_{key}"] = float(pair_correct / pair_valid) if pair_valid > 0 else float("nan")

            if regrets:
                regret_tensor = torch.stack(regrets).float()
                normalized_regret_tensor = torch.stack(normalized_regrets).float()
                top1_tensor = torch.stack(top1_hits).float()
                zero_regret_tensor = torch.stack(zero_regret_hits).float()
                metrics[f"selection_mean_regret_{key}"] = float(regret_tensor.mean().item())
                metrics[f"selection_median_regret_{key}"] = float(regret_tensor.median().item())
                metrics[f"selection_normalized_mean_regret_{key}"] = float(normalized_regret_tensor.mean().item())
                metrics[f"selection_top1_setting_accuracy_{key}"] = float(top1_tensor.mean().item())
                metrics[f"selection_zero_regret_ratio_{key}"] = float(zero_regret_tensor.mean().item())
            else:
                metrics[f"selection_mean_regret_{key}"] = float("nan")
                metrics[f"selection_median_regret_{key}"] = float("nan")
                metrics[f"selection_normalized_mean_regret_{key}"] = float("nan")
                metrics[f"selection_top1_setting_accuracy_{key}"] = float("nan")
                metrics[f"selection_zero_regret_ratio_{key}"] = float("nan")

    return {
        key: metrics[key]
        for key in logged_metric_names
        if key in metrics
    }

# @torch.no_grad()
# def compute_masked_correlations(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     valid_mask: torch.Tensor,
#     max_samples: Optional[int] = 100_000,
#     prefix: str = "correlation",
# ) -> Dict[str, float]:
#     """
#     Compute Pearson and Spearman correlations between two per-pixel maps.

#     The valid pixels are optionally sub-sampled with deterministic uniform
#     indexing so this metric does not perturb the training RNG state.
#     """
#     if x.shape != y.shape:
#         raise ValueError(f"Shape mismatch: x {tuple(x.shape)} != y {tuple(y.shape)}")

#     x_flat, y_flat = prepare_masked_vectors(x, y, valid_mask, max_samples=max_samples)
#     if x_flat.numel() < 2:
#         return {
#             f"{prefix}_pearson": float("nan"),
#             f"{prefix}_spearman": float("nan"),
#         }

#     pearson = pearson_corr(x_flat, y_flat)
#     spearman = spearman_corr(x_flat, y_flat)
#     return {
#         f"{prefix}_pearson": float(pearson.item()),
#         f"{prefix}_spearman": float(spearman.item()),
#     }
