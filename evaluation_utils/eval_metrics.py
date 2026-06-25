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
    calc_dtype = torch.float64 if mu.dtype == torch.float64 or target.dtype == torch.float64 else torch.float32
    pred = mu.to(dtype=calc_dtype)
    gt = target.to(dtype=calc_dtype)
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
    num_bins: int = 100
) -> Dict[str, torch.Tensor]:
    """
    Compute MDE-style AUSE metrics by removing high-uncertainty pixels first.

    ``ause_abs_rel`` uses per-pixel AbsRel error. ``ause_a1`` uses the a1
    failure indicator, so lower AUSE is better for both metrics.
    """
    
    abs_rel_error, a1_error = depth_error_maps(mu, target)
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
) -> Dict[str, torch.Tensor]:
    """
    Compute MDE-style AURG metrics by removing high-uncertainty pixels first.

    ``aurg_abs_rel`` uses per-pixel AbsRel error. ``aurg_a1`` uses the a1
    failure indicator, so higher AURG is better for both metrics.
    """
    abs_rel_error, a1_error = depth_error_maps(mu, target)
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
) -> Dict[str, torch.Tensor]:
    gt = ensure_bchw(gt)
    valid_mask = prepare_bchw_mask(valid_mask, gt)
    l1_loss_map = calculate_l1_depth_lossmap(mu, gt, valid_mask)
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
    log_var: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: Optional[torch.Tensor] = None,
    max_samples: Optional[int] = 100_000
) -> Dict[str, torch.Tensor]:
    """
    Compute correlation between per-pixel L1 depth loss and uncertainty.

    Args:
        uncertainty_kind:
            Used only when ``uncertainty`` is None. One of "std", "var", or
            "log_var".
    """
    loss_map = calculate_l1_depth_lossmap(mu, target, valid_mask)
    if uncertainty is None:
        uncertainty = torch.exp(0.5 * log_var)
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
    