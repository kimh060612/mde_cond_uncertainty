from __future__ import annotations

import torch
import torch.nn.functional as F

def gaussian_nll_depth_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_smooth_logvar: float = 0.0,
) -> torch.Tensor:
    """
    mu:        [B, 1, H, W]
    log_var:   [B, 1, H, W]
    target:    [B, H, W]
    valid_mask:[B, H, W]
    """
    target = target.unsqueeze(1)
    valid_mask = valid_mask.unsqueeze(1)

    residual2 = (target - mu) ** 2
    nll = 0.5 * (torch.exp(-log_var) * residual2 + log_var)

    nll = nll * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    loss = nll.sum() / denom

    if lambda_smooth_logvar > 0.0:
        dx = torch.abs(log_var[:, :, :, 1:] - log_var[:, :, :, :-1]).mean()
        dy = torch.abs(log_var[:, :, 1:, :] - log_var[:, :, :-1, :]).mean()
        loss = loss + lambda_smooth_logvar * (dx + dy)

    return loss


def faithful_heteroscedastic_depth_loss(
    mu: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_smooth_logvar: float = 0.0,
):
    """
    Separate mean regression from variance regression.

    The mean loss updates only ``mu``. The variance NLL uses a detached mean
    residual, so variance learning cannot hide mean errors by pushing the mean.
    """
    target = target.unsqueeze(1)
    valid_mask = valid_mask.unsqueeze(1).bool()

    mean_loss = F.smooth_l1_loss(
        mu[valid_mask],
        target[valid_mask],
    )

    safe_variance = variance.clamp_min(1e-8)
    detached_residual2 = (target - mu.detach()).square()
    variance_nll = 0.5 * (
        detached_residual2 / safe_variance
        + torch.log(safe_variance)
    )
    variance_loss = variance_nll[valid_mask].mean()

    if lambda_smooth_logvar > 0.0:
        log_variance = torch.log(safe_variance)
        dx = torch.abs(log_variance[:, :, :, 1:] - log_variance[:, :, :, :-1]).mean()
        dy = torch.abs(log_variance[:, :, 1:, :] - log_variance[:, :, :-1, :]).mean()
        variance_loss = variance_loss + lambda_smooth_logvar * (dx + dy)

    return mean_loss, variance_loss

def fheteroscedastic_caminduced_depth_loss(
    corrected_depth: torch.Tensor,
    variance: torch.Tensor,
    canonical_depth: torch.Tensor,
    lambda_smooth_logvar: float = 0.0,
):
    """
    Separate mean regression from variance regression.

    The mean loss updates only ``mu``. The variance NLL uses a detached mean
    residual, so variance learning cannot hide mean errors by pushing the mean.
    """
    mean_loss = F.smooth_l1_loss(corrected_depth, canonical_depth)
    
    safe_variance = variance.clamp_min(1e-8)
    detached_residual2 = (canonical_depth - corrected_depth.detach()).square()
    variance_nll = 0.5 * (detached_residual2 / safe_variance + torch.log(safe_variance))
    variance_loss = variance_nll.mean()

    if lambda_smooth_logvar > 0.0:
        log_variance = torch.log(safe_variance)
        dx = torch.abs(log_variance[:, :, :, 1:] - log_variance[:, :, :, :-1]).mean()
        dy = torch.abs(log_variance[:, :, 1:, :] - log_variance[:, :, :-1, :]).mean()
        variance_loss = variance_loss + lambda_smooth_logvar * (dx + dy)

    return mean_loss, variance_loss


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected [B, H, W] or [B, 1, H, W], got {tuple(x.shape)}")


def scale_shift_invariant_depth_loss(
    candidate_depth: torch.Tensor,
    canonical_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Image-level scale-shift invariant L1 distance between two relative depth maps.
    The candidate map is linearly aligned to the canonical map per image.
    """
    candidate_depth = _ensure_bchw(candidate_depth).float()
    canonical_depth = _ensure_bchw(canonical_depth).float()
    if candidate_depth.shape != canonical_depth.shape:
        raise ValueError("candidate_depth and canonical_depth must have the same shape.")

    if valid_mask is None:
        mask = torch.ones_like(candidate_depth, dtype=torch.bool)
    else:
        mask = _ensure_bchw(valid_mask).bool()
        if mask.shape != candidate_depth.shape:
            mask = mask.expand_as(candidate_depth)

    mask = mask & torch.isfinite(candidate_depth) & torch.isfinite(canonical_depth)
    mask = mask & (candidate_depth > 0) & (canonical_depth > 0)
    mask_f = mask.to(candidate_depth.dtype)

    valid_counts = mask_f.flatten(1).sum(dim=1)
    safe_counts = valid_counts.clamp_min(1.0)
    x = torch.where(mask, candidate_depth, torch.zeros_like(candidate_depth))
    y = torch.where(mask, canonical_depth, torch.zeros_like(canonical_depth))

    sum_x = x.flatten(1).sum(dim=1)
    sum_y = y.flatten(1).sum(dim=1)
    sum_xx = (x * x).flatten(1).sum(dim=1)
    sum_xy = (x * y).flatten(1).sum(dim=1)

    denom = safe_counts * sum_xx - sum_x.square()
    stable = (valid_counts > 1) & (denom.abs() > eps)
    scale = torch.where(
        stable,
        (safe_counts * sum_xy - sum_x * sum_y) / denom.clamp_min(eps),
        torch.zeros_like(valid_counts),
    )
    shift = torch.where(
        valid_counts > 0,
        (sum_y - scale * sum_x) / safe_counts,
        torch.zeros_like(valid_counts),
    )

    aligned_candidate = scale.view(-1, 1, 1, 1) * candidate_depth + shift.view(-1, 1, 1, 1)
    loss_map = torch.abs(aligned_candidate - canonical_depth)
    loss = (loss_map * mask_f).flatten(1).sum(dim=1) / safe_counts
    return torch.where(valid_counts > 0, loss, loss.new_full(loss.shape, float("nan")))

import torch


def log_scale_invariant_depth_difference(
    candidate_depth: torch.Tensor,
    canonical_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    # Replace these two calls with your existing _ensure_bchw implementation.
    candidate_depth = _ensure_bchw(candidate_depth).float()
    canonical_depth = _ensure_bchw(canonical_depth).float()
    if valid_mask is None:
        mask = torch.ones_like(candidate_depth, dtype=torch.bool)
    else:
        mask = _ensure_bchw(valid_mask).bool()
        mask = mask.expand_as(candidate_depth)
        
    # Only finite, strictly positive depth values are valid in the log domain.
    mask = (
        mask
        & torch.isfinite(candidate_depth)
        & torch.isfinite(canonical_depth)
        & (candidate_depth > 0)
        & (canonical_depth > 0)
    )

    batch_size = candidate_depth.shape[0]
    flat_mask = mask.flatten(1)
    valid_counts = flat_mask.sum(dim=1)
    safe_counts = valid_counts.clamp_min(1).to(candidate_depth.dtype)

    # eps guards against numerical issues near zero. Values that were actually
    # non-positive have already been removed by the validity mask.
    candidate_log = torch.log(candidate_depth.clamp_min(eps))
    canonical_log = torch.log(canonical_depth.clamp_min(eps))

    # log(candidate / canonical)
    log_ratio = candidate_log - canonical_log
    flat_log_ratio = log_ratio.flatten(1)

    # NaN masking allows a per-image median over only valid pixels.
    masked_log_ratio = torch.where(
        flat_mask,
        flat_log_ratio,
        torch.full_like(flat_log_ratio, float("nan")),
    )

    # Shape: [B]
    global_log_scale = torch.nanmedian(masked_log_ratio, dim=1).values

    # Remove the global multiplicative scale difference.
    centered_log_ratio = (
        log_ratio
        - global_log_scale.view(batch_size, 1, 1, 1)
    )

    absolute_difference = torch.abs(centered_log_ratio)

    loss = (
        torch.where(
            mask,
            absolute_difference,
            torch.zeros_like(absolute_difference),
        )
        .flatten(1)
        .sum(dim=1)
        / safe_counts
    )

    return torch.where(
        valid_counts > 0,
        loss,
        loss.new_full(loss.shape, float("nan")),
    )

def sobel_log_gradient_depth_difference(
    candidate_depth: torch.Tensor,
    canonical_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    candidate_depth = _ensure_bchw(candidate_depth).float()
    canonical_depth = _ensure_bchw(canonical_depth).float()

    if candidate_depth.shape != canonical_depth.shape:
        raise ValueError(
            "candidate_depth and canonical_depth must have the same shape."
        )

    if valid_mask is None:
        mask = torch.ones_like(candidate_depth, dtype=torch.bool)
    else:
        mask = _ensure_bchw(valid_mask).bool()
        try:
            mask = mask.expand_as(candidate_depth)
        except RuntimeError as exc:
            raise ValueError(
                "valid_mask must be broadcast-compatible with the depth maps."
            ) from exc

    mask = (
        mask
        & torch.isfinite(candidate_depth)
        & torch.isfinite(canonical_depth)
        & (candidate_depth > 0)
        & (canonical_depth > 0)
    )

    candidate_log = torch.log(candidate_depth.clamp_min(eps))
    canonical_log = torch.log(canonical_depth.clamp_min(eps))

    channels = candidate_depth.shape[1]

    sobel_x = candidate_depth.new_tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]
    ).view(1, 1, 3, 3) / 8.0

    sobel_y = candidate_depth.new_tensor(
        [
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0],
        ]
    ).view(1, 1, 3, 3) / 8.0

    # Apply the same Sobel kernel independently to every channel.
    sobel_x = sobel_x.expand(channels, 1, 3, 3)
    sobel_y = sobel_y.expand(channels, 1, 3, 3)

    candidate_gx = F.conv2d(
        candidate_log,
        sobel_x,
        padding=1,
        groups=channels,
    )
    candidate_gy = F.conv2d(
        candidate_log,
        sobel_y,
        padding=1,
        groups=channels,
    )

    canonical_gx = F.conv2d(
        canonical_log,
        sobel_x,
        padding=1,
        groups=channels,
    )
    canonical_gy = F.conv2d(
        canonical_log,
        sobel_y,
        padding=1,
        groups=channels,
    )

    # A Sobel output is valid only if all pixels in its 3x3 neighborhood
    # are valid. Grouped convolution handles each channel independently.
    validity_kernel = candidate_depth.new_ones(
        (channels, 1, 3, 3)
    )

    valid_neighbor_counts = F.conv2d(
        mask.to(candidate_depth.dtype),
        validity_kernel,
        padding=1,
        groups=channels,
    )

    gradient_mask = valid_neighbor_counts == 9.0

    difference_map = (
        torch.abs(candidate_gx - canonical_gx)
        + torch.abs(candidate_gy - canonical_gy)
    )

    difference_sum = torch.where(
        gradient_mask,
        difference_map,
        torch.zeros_like(difference_map),
    ).flatten(1).sum(dim=1)

    valid_counts = gradient_mask.flatten(1).sum(dim=1)
    safe_counts = valid_counts.clamp_min(1).to(candidate_depth.dtype)

    difference = difference_sum / safe_counts

    return torch.where(
        valid_counts > 0,
        difference,
        difference.new_full(difference.shape, float("nan")),
    )


def sobel_log_gradient_magnitude_difference(
    candidate_depth: torch.Tensor,
    canonical_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    candidate_depth = _ensure_bchw(candidate_depth).float()
    canonical_depth = _ensure_bchw(canonical_depth).float()

    if candidate_depth.shape != canonical_depth.shape:
        raise ValueError(
            "candidate_depth and canonical_depth must have the same shape."
        )

    if valid_mask is None:
        mask = torch.ones_like(candidate_depth, dtype=torch.bool)
    else:
        mask = _ensure_bchw(valid_mask).bool()
        try:
            mask = mask.expand_as(candidate_depth)
        except RuntimeError as exc:
            raise ValueError(
                "valid_mask must be broadcast-compatible with the depth maps."
            ) from exc

    mask = (
        mask
        & torch.isfinite(candidate_depth)
        & torch.isfinite(canonical_depth)
        & (candidate_depth > 0)
        & (canonical_depth > 0)
    )

    candidate_log = torch.log(candidate_depth.clamp_min(eps))
    canonical_log = torch.log(canonical_depth.clamp_min(eps))

    channels = candidate_depth.shape[1]

    sobel_x = candidate_depth.new_tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]
    ).view(1, 1, 3, 3) / 8.0

    sobel_y = candidate_depth.new_tensor(
        [
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0],
        ]
    ).view(1, 1, 3, 3) / 8.0

    sobel_x = sobel_x.expand(channels, 1, 3, 3)
    sobel_y = sobel_y.expand(channels, 1, 3, 3)

    candidate_gx = F.conv2d(
        candidate_log,
        sobel_x,
        padding=1,
        groups=channels,
    )
    candidate_gy = F.conv2d(
        candidate_log,
        sobel_y,
        padding=1,
        groups=channels,
    )

    canonical_gx = F.conv2d(
        canonical_log,
        sobel_x,
        padding=1,
        groups=channels,
    )
    canonical_gy = F.conv2d(
        canonical_log,
        sobel_y,
        padding=1,
        groups=channels,
    )

    candidate_magnitude = torch.sqrt(
        candidate_gx.square() + candidate_gy.square() + eps
    )
    canonical_magnitude = torch.sqrt(
        canonical_gx.square() + canonical_gy.square() + eps
    )

    validity_kernel = candidate_depth.new_ones(
        (channels, 1, 3, 3)
    )

    valid_neighbor_counts = F.conv2d(
        mask.to(candidate_depth.dtype),
        validity_kernel,
        padding=1,
        groups=channels,
    )

    gradient_mask = valid_neighbor_counts == 9.0

    difference_map = torch.abs(
        candidate_magnitude - canonical_magnitude
    )

    difference_sum = torch.where(
        gradient_mask,
        difference_map,
        torch.zeros_like(difference_map),
    ).flatten(1).sum(dim=1)

    valid_counts = gradient_mask.flatten(1).sum(dim=1)
    safe_counts = valid_counts.clamp_min(1).to(candidate_depth.dtype)

    difference = difference_sum / safe_counts

    return torch.where(
        valid_counts > 0,
        difference,
        difference.new_full(difference.shape, float("nan")),
    )

def scalar_heteroscedastic_loss(
    predicted_mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_mean = predicted_mean.flatten()
    variance = variance.flatten()
    target = target.detach().flatten().to(dtype=predicted_mean.dtype)

    valid_mask = (
        torch.isfinite(predicted_mean)
        & torch.isfinite(variance)
        & torch.isfinite(target)
    )
    if not valid_mask.any():
        zero = predicted_mean.sum() * 0.0 + variance.sum() * 0.0
        return zero, zero

    mean_loss = F.smooth_l1_loss(predicted_mean[valid_mask], target[valid_mask])
    safe_variance = variance[valid_mask].clamp_min(1e-8)
    detached_residual2 = (target[valid_mask] - predicted_mean[valid_mask].detach()).square()
    variance_loss = 0.5 * (detached_residual2 / safe_variance + torch.log(safe_variance))
    return mean_loss, variance_loss.mean()

def scalar_heteroscedastic_laplace_loss(
    predicted_mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_mean = predicted_mean.flatten()
    variance = variance.flatten()
    target = target.detach().flatten().to(dtype=predicted_mean.dtype)
    valid_mask = (
        torch.isfinite(predicted_mean)
        & torch.isfinite(variance)
        & torch.isfinite(target)
    )
    if not valid_mask.any():
        zero = predicted_mean.sum() * 0.0 + variance.sum() * 0.0
        return zero, zero

    mean_loss = F.l1_loss(predicted_mean[valid_mask], target[valid_mask])
    safe_variance = variance[valid_mask].clamp_min(1e-8)
    detached_residual = (target[valid_mask] - predicted_mean[valid_mask].detach()).abs()
    variance_loss = detached_residual / safe_variance + torch.log(safe_variance)
    return mean_loss, variance_loss.mean()

def image_absrel_error(mu, depth, valid_mask):
    depth = depth.unsqueeze(1)
    mask = valid_mask.unsqueeze(1).bool()

    errors = []

    for b in range(mu.shape[0]):
        pred = mu[b][mask[b]].clamp_min(1e-3)
        gt = depth[b][mask[b]].clamp_min(1e-3)

        if pred.numel() == 0:
            errors.append(torch.tensor(0.0, device=mu.device))
        else:
            errors.append(torch.mean(torch.abs(pred - gt) / gt))

    return torch.stack(errors)


def image_uncertainty_score(std, valid_mask, mode="top20"):
    if valid_mask.ndim == 3:
        mask = valid_mask.unsqueeze(1).bool()
    else:
        mask = valid_mask.bool()

    scores = []

    for b in range(std.shape[0]):
        u = std[b][mask[b]]

        if u.numel() == 0:
            scores.append(torch.tensor(0.0, device=std.device))
            continue

        if mode == "mean":
            score = u.mean()
        elif mode == "top20":
            k = max(1, int(0.20 * u.numel()))
            score = torch.topk(u, k=k, largest=True).values.mean()
        elif mode == "top10":
            k = max(1, int(0.10 * u.numel()))
            score = torch.topk(u, k=k, largest=True).values.mean()
        else:
            raise ValueError(mode)

        scores.append(score)

    return torch.stack(scores)

def image_level_listnet_loss(
    mu,
    std,
    depth,
    valid_mask,
    temperature=0.1,
    uncertainty_mode="top20",
):
    with torch.no_grad():
        image_error = image_absrel_error(mu.detach(), depth, valid_mask)

    image_unc = image_uncertainty_score(std, valid_mask, mode=uncertainty_mode)

    if image_error.numel() < 2:
        return torch.tensor(0.0, device=mu.device)

    # normalize for stability
    e = image_error
    u = image_unc

    e = (e - e.mean()) / (e.std().clamp_min(1e-6))
    u = (u - u.mean()) / (u.std().clamp_min(1e-6))

    target_prob = F.softmax(e / temperature, dim=0).detach()
    pred_log_prob = F.log_softmax(u / temperature, dim=0)

    loss = -(target_prob * pred_log_prob).sum()
    return loss



def image_level_ranknet_loss(
    mu,
    std,
    depth,
    valid_mask,
    min_error_gap=0.005,
    temperature=0.5,
    uncertainty_mode="top20",
):
    """
    Batch-level image ranking:
    images with larger depth error should have larger aggregated uncertainty.
    """
    with torch.no_grad():
        image_error = image_absrel_error(mu.detach(), depth, valid_mask)

    image_unc = image_uncertainty_score(std, valid_mask, mode=uncertainty_mode)

    B = image_error.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=mu.device)

    losses = []

    for i in range(B):
        for j in range(B):
            if i == j:
                continue

            diff_e = image_error[i] - image_error[j]
            if torch.abs(diff_e) <= min_error_gap:
                continue

            sign = torch.sign(diff_e)
            diff_u = image_unc[i] - image_unc[j]

            loss = F.softplus(-sign * diff_u / temperature)
            losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=mu.device)

    return torch.stack(losses).mean()



def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: tuple[int, ...],
    eps: float = 1e-6,
) -> torch.Tensor:
    mask_f = mask.to(values.dtype)

    numerator = (values * mask_f).sum(dim=dim)
    denominator = mask_f.sum(dim=dim).clamp_min(eps)
    return numerator / denominator


def camera_risk_scores(
    bias: torch.Tensor,
    variance: torch.Tensor,
    canonical_delta: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        bias:
            [G, K, 1, H, W]
        variance:
            [G, K, 1, H, W]
        canonical_delta:
            [G, K, 1, H, W]
            canonical prediction - candidate prediction
        valid_mask:
            [G, K, 1, H, W]

        G:
            Number of scene/context groups.

        K:
            Number of candidate camera parameters per group.

    Returns:
        predicted_risk:
            [G, K], mean_p[b^2 + variance]

        target_risk:
            [G, K], mean_p[canonical_delta^2]
    """
    spatial_dims = (2, 3, 4)

    predicted_risk = masked_mean(
        bias.square() + variance,
        valid_mask,
        dim=spatial_dims,
        eps=eps,
    )

    with torch.no_grad():
        target_risk = masked_mean(
            canonical_delta.square(),
            valid_mask,
            dim=spatial_dims,
            eps=eps,
        )

    return (
        predicted_risk.clamp_min(eps),
        target_risk.clamp_min(eps),
    )

def gap_weighted_ranknet_loss(
    predicted_risk: torch.Tensor,
    target_risk: torch.Tensor,
    temperature: float = 0.5,
    tie_margin: float = 0.05,
    max_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Args:
        predicted_risk:
            [G, K]. Higher means worse camera parameter.

        target_risk:
            [G, K]. GT-free canonical pseudo-risk.
            Higher means worse camera parameter.

    Returns:
        Scalar pairwise ranking loss.
    """
    if predicted_risk.shape != target_risk.shape:
        raise ValueError("predicted_risk and target_risk must have the same shape.")

    _, num_candidates = predicted_risk.shape

    if num_candidates < 2:
        return predicted_risk.new_zeros(())

    pred_score = torch.log(predicted_risk + eps)
    target_score = torch.log(target_risk.detach() + eps)

    # All upper-triangular candidate pairs.
    pair_i, pair_j = torch.triu_indices(num_candidates, num_candidates, offset=1, device=predicted_risk.device)
    pred_diff = pred_score[:, pair_i] - pred_score[:, pair_j]
    target_diff = target_score[:, pair_i] - target_score[:, pair_j]

    # +1 means candidate i should have higher predicted risk.
    pair_label = torch.sign(target_diff)
    normalized_gap = (target_risk[:, pair_i] - target_risk[:, pair_j]).abs() / (target_risk[:, pair_i] + target_risk[:, pair_j] + eps)
    valid_pairs = (
        normalized_gap >= tie_margin
    ) & (pair_label != 0)

    if not valid_pairs.any():
        return predicted_risk.new_zeros(())

    pair_weights = normalized_gap.clamp(max=max_weight)
    pair_loss = F.softplus(-pair_label * pred_diff / temperature)
    weighted_loss = pair_loss[valid_pairs] * pair_weights[valid_pairs]
    return weighted_loss.sum() / pair_weights[valid_pairs].sum().clamp_min(eps)


def signed_pairwise_ranknet_loss(
    predicted_score: torch.Tensor,
    target_score: torch.Tensor,
    temperature: float = 0.5,
    tie_margin: float = 0.0,
    max_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pairwise ranking loss for signed scores such as AbsRel degradation.
    Higher predicted_score should mean higher target_score.
    """
    if predicted_score.shape != target_score.shape:
        raise ValueError("predicted_score and target_score must have the same shape.")

    _, num_candidates = predicted_score.shape
    if num_candidates < 2:
        return predicted_score.new_zeros(())

    pair_i, pair_j = torch.triu_indices(
        num_candidates,
        num_candidates,
        offset=1,
        device=predicted_score.device,
    )
    pred_diff = predicted_score[:, pair_i] - predicted_score[:, pair_j]
    target_diff = target_score[:, pair_i].detach() - target_score[:, pair_j].detach()
    pair_label = torch.sign(target_diff)
    gap = target_diff.abs()
    valid_pairs = (gap > tie_margin) & (pair_label != 0)

    if not valid_pairs.any():
        return predicted_score.new_zeros(())

    normalizer = (
        target_score[:, pair_i].detach().abs()
        + target_score[:, pair_j].detach().abs()
        + eps
    )
    pair_weights = (gap / normalizer).clamp(max=max_weight)
    pair_loss = F.softplus(-pair_label * pred_diff / max(temperature, eps))
    weighted_loss = pair_loss[valid_pairs] * pair_weights[valid_pairs]
    return weighted_loss.sum() / pair_weights[valid_pairs].sum().clamp_min(eps)


def listwise_camera_ranking_loss(
    predicted_risk: torch.Tensor,
    target_risk: torch.Tensor,
    temperature: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_score = torch.log(predicted_risk + eps)
    target_score = torch.log(target_risk.detach() + eps)

    target_preferences = F.softmax(-target_score / temperature, dim=1)
    predicted_log_preferences = F.log_softmax(-pred_score / temperature, dim=1)

    return -(target_preferences * predicted_log_preferences).sum(dim=1).mean()
