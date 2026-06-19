from typing import Dict, Optional, Tuple

import torch


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected tensor with shape [B, H, W] or [B, 1, H, W], got {tuple(x.shape)}")


def _cpu_float_bchw(x: torch.Tensor) -> torch.Tensor:
    return _ensure_bchw(x.detach()).float().cpu()


def _cpu_bool_bchw(x: torch.Tensor) -> torch.Tensor:
    return _ensure_bchw(x.detach()).bool().cpu()


def _prepare_bchw_mask(valid_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = _ensure_bchw(mu).clamp_min(1e-3)
    target = _ensure_bchw(target).clamp_min(1e-3)

    abs_rel_error = torch.abs(mu - target) / target
    ratio = torch.maximum(mu / target, target / mu)
    a1_error = (ratio >= 1.25).float()

    return abs_rel_error, a1_error


def gaussian_nll_loss_map(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Return the per-pixel Gaussian NLL data term without reduction.

    Shapes:
        mu:      [B, 1, H, W]
        log_var: [B, 1, H, W]
        target:  [B, H, W] or [B, 1, H, W]
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)

    residual2 = (target - mu) ** 2
    return 0.5 * (torch.exp(-log_var) * residual2 + log_var)


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
    x = _cpu_float_bchw(x)
    y = _cpu_float_bchw(y)
    valid_mask = _cpu_bool_bchw(valid_mask)

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x {tuple(x.shape)} != y {tuple(y.shape)}")

    x_flat, y_flat = _prepare_masked_vectors(x, y, valid_mask, max_samples=max_samples)

    if x_flat.numel() < 2:
        return {
            f"{prefix}_pearson": float("nan"),
            f"{prefix}_spearman": float("nan"),
            f"{prefix}_samples": int(x_flat.numel()),
        }

    pearson = _pearson_corr(x_flat, y_flat)
    spearman = _spearman_corr(x_flat, y_flat)

    return {
        f"{prefix}_pearson": float(pearson.item()),
        f"{prefix}_spearman": float(spearman.item()),
        f"{prefix}_samples": int(x_flat.numel()),
    }


@torch.no_grad()
def compute_loss_uncertainty_correlations(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: Optional[torch.Tensor] = None,
    uncertainty_kind: str = "std",
    max_samples: Optional[int] = 100_000,
) -> Dict[str, float]:
    """
    Compute correlation between per-pixel Gaussian NLL loss and uncertainty.

    Args:
        uncertainty_kind:
            Used only when ``uncertainty`` is None. One of "std", "var", or
            "log_var".
    """
    mu = _cpu_float_bchw(mu)
    log_var = _cpu_float_bchw(log_var)
    target = _cpu_float_bchw(target)
    valid_mask = _cpu_bool_bchw(valid_mask)
    loss_map = gaussian_nll_loss_map(mu, log_var, target)

    if uncertainty is None:
        if uncertainty_kind == "std":
            uncertainty = torch.exp(0.5 * log_var)
        elif uncertainty_kind == "var":
            uncertainty = torch.exp(log_var)
        elif uncertainty_kind == "log_var":
            uncertainty = log_var
        else:
            raise ValueError(f"Unsupported uncertainty_kind: {uncertainty_kind}")
    else:
        uncertainty = _cpu_float_bchw(uncertainty)

    return compute_masked_correlations(
        loss_map,
        uncertainty,
        valid_mask,
        max_samples=max_samples,
        prefix="loss_uncertainty",
    )


def _sparsification_ause(
    error: torch.Tensor,
    uncertainty: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    if error.numel() < 2:
        return error.new_tensor(float("nan"))

    error = error.float()
    uncertainty = uncertainty.float()
    n = error.numel()
    num_points = min(num_bins + 1, n)

    remove_counts = torch.linspace(
        0,
        n - 1,
        steps=num_points,
        device=error.device,
    ).round().long()
    remove_counts = torch.unique_consecutive(remove_counts)
    if remove_counts.numel() < 2:
        return error.new_tensor(float("nan"))

    uncertainty_order = torch.argsort(uncertainty, descending=True)
    oracle_order = torch.argsort(error, descending=True)
    uncertainty_sorted_error = error[uncertainty_order]
    oracle_sorted_error = error[oracle_order]

    uncertainty_curve = torch.stack(
        [uncertainty_sorted_error[remove_count:].mean() for remove_count in remove_counts]
    )
    oracle_curve = torch.stack(
        [oracle_sorted_error[remove_count:].mean() for remove_count in remove_counts]
    )
    removed_fractions = remove_counts.double() / float(n)

    return torch.trapz((uncertainty_curve - oracle_curve).double(), removed_fractions)


@torch.no_grad()
def compute_sparsification_ause_metrics(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    max_samples: Optional[int] = 100_000,
    num_bins: int = 100,
) -> Dict[str, float]:
    """
    Compute MDE-style AUSE metrics by removing high-uncertainty pixels first.

    ``ause_abs_rel`` uses per-pixel AbsRel error. ``ause_a1`` uses the a1
    failure indicator, so lower AUSE is better for both metrics.
    """
    mu = _cpu_float_bchw(mu)
    target = _cpu_float_bchw(target)
    uncertainty = _cpu_float_bchw(uncertainty)
    mask = _prepare_bchw_mask(_cpu_bool_bchw(valid_mask), mu)

    abs_rel_error, a1_error = _depth_error_maps(mu, target)
    per_image_max_samples = max_samples
    if max_samples is not None and max_samples > 0:
        per_image_max_samples = max(1, max_samples // max(mu.shape[0], 1))

    ause_abs_rel = []
    ause_a1 = []
    total_samples = 0

    for batch_idx in range(mu.shape[0]):
        finite_mask = (
            mask[batch_idx]
            & torch.isfinite(abs_rel_error[batch_idx])
            & torch.isfinite(a1_error[batch_idx])
            & torch.isfinite(uncertainty[batch_idx])
        )

        abs_rel_flat = abs_rel_error[batch_idx][finite_mask].flatten()
        a1_flat = a1_error[batch_idx][finite_mask].flatten()
        uncertainty_flat = uncertainty[batch_idx][finite_mask].flatten()

        abs_rel_flat, a1_flat, uncertainty_flat = _deterministic_subsample(
            abs_rel_flat,
            a1_flat,
            uncertainty_flat,
            max_samples=per_image_max_samples,
        )

        if uncertainty_flat.numel() < 2:
            continue

        total_samples += int(uncertainty_flat.numel())
        ause_abs_rel.append(_sparsification_ause(abs_rel_flat, uncertainty_flat, num_bins=num_bins))
        ause_a1.append(_sparsification_ause(a1_flat, uncertainty_flat, num_bins=num_bins))

    if not ause_abs_rel:
        return {
            "ause_abs_rel": float("nan"),
            "ause_a1": float("nan"),
            "ause_samples": int(total_samples),
        }

    return {
        "ause_abs_rel": float(torch.stack(ause_abs_rel).mean().item()),
        "ause_a1": float(torch.stack(ause_a1).mean().item()),
        "ause_samples": int(total_samples),
    }


@torch.no_grad()
def compute_image_uncertainty_metric_values(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    uncertainty: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Return per-image mean uncertainty and per-image depth metrics.
    """
    mu = _cpu_float_bchw(mu)
    target = _cpu_float_bchw(target)
    uncertainty = _cpu_float_bchw(uncertainty)
    mask = _prepare_bchw_mask(_cpu_bool_bchw(valid_mask), mu)

    abs_rel_error, a1_error = _depth_error_maps(mu, target)
    mean_uncertainty = []
    abs_rel = []
    a1 = []

    for batch_idx in range(mu.shape[0]):
        finite_mask = (
            mask[batch_idx]
            & torch.isfinite(abs_rel_error[batch_idx])
            & torch.isfinite(a1_error[batch_idx])
            & torch.isfinite(uncertainty[batch_idx])
        )

        if finite_mask.sum() == 0:
            continue

        mean_uncertainty.append(uncertainty[batch_idx][finite_mask].float().mean())
        abs_rel.append(abs_rel_error[batch_idx][finite_mask].float().mean())
        a1.append(1.0 - a1_error[batch_idx][finite_mask].float().mean())

    if not mean_uncertainty:
        empty = mu.new_empty(0, dtype=torch.float32)
        return {
            "mean_uncertainty": empty,
            "abs_rel": empty,
            "a1": empty,
        }

    return {
        "mean_uncertainty": torch.stack(mean_uncertainty).float(),
        "abs_rel": torch.stack(abs_rel).float(),
        "a1": torch.stack(a1).float(),
    }


@torch.no_grad()
def compute_image_uncertainty_metric_correlations(
    image_values: Dict[str, torch.Tensor],
    prefix: str = "image_mean_uncertainty",
) -> Dict[str, float]:
    """
    Correlate image-level mean uncertainty with image-level depth metrics.
    """
    mean_uncertainty = image_values["mean_uncertainty"].detach().float().cpu()
    metrics = {
        "abs_rel": image_values["abs_rel"].detach().float().cpu(),
        "a1": image_values["a1"].detach().float().cpu(),
    }

    result = {
        prefix: float(mean_uncertainty.mean().item()) if mean_uncertainty.numel() > 0 else float("nan"),
        f"{prefix}_samples": int(mean_uncertainty.numel()),
    }

    for metric_name, metric_values in metrics.items():
        finite_mask = torch.isfinite(mean_uncertainty) & torch.isfinite(metric_values)
        x = mean_uncertainty[finite_mask]
        y = metric_values[finite_mask]

        if x.numel() < 2:
            pearson = x.new_tensor(float("nan"))
            spearman = x.new_tensor(float("nan"))
        else:
            pearson = _pearson_corr(x, y)
            spearman = _spearman_corr(x, y)

        result[f"{prefix}_{metric_name}_pearson"] = float(pearson.item())
        result[f"{prefix}_{metric_name}_spearman"] = float(spearman.item())

    return result
