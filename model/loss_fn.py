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

