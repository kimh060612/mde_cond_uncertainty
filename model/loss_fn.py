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