from typing import Tuple, Dict
import torch
from model.loss_fn import gaussian_nll_depth_loss, image_level_listnet_loss
from evaluation_utils.eval_utils import (
    align_relative_prediction_to_depth_space,
    ensure_bchw,
    _mean_finite_metrics,
)
from evaluation_utils.eval_metrics import (
    compute_sparsification_ause_metrics,
    compute_sparsification_aurg_metrics,
    compute_comprehensive_depth_metrics,
    compute_loss_uncertainty_correlations
)
from tqdm.auto import tqdm
from utils.train_utils import *
# from utils.logger import wandb_log_prefixed
import logging

def train_one_epoch(
    model_id: str,
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    amp: bool,
    lambda_smooth_logvar: float,
    list_loss_weight: float,
    listnet_temperature: float,
    uncertainty_mode: str,
    grad_clip: float,
    logger: logging.Logger,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    correlation_max_samples: int = 100_000,
    relative_align_mode: str = "scale_shift",
    global_step: int = 0,
    log_interval: int = 20,
)-> Tuple[Dict[str, float], int]:
    model.train()
    progress_bar = tqdm(
        loader,
        desc=f"Train {epoch:03d}",
        dynamic_ncols=True,
        leave=False,
    )
    
    running_loss = 0.0
    running_nll_loss = 0.0
    running_list_loss = 0.0
    running_abs_rel = 0.0
    running_rmse = 0.0
    running_a1 = 0.0
    running_ause_abs_rel = []
    running_aurg_abs_rel = []
    running_ause_a1 = []
    running_aurg_a1 = []
    running_pearson_correlation_l1 = []
    running_spearman_correlation_l1 = []
    corr_sums = {}
    corr_counts = {}
    condition_sums = {}
    processed_batches = 0

    for step, batch in enumerate(progress_bar, start=1):
        if batch is None:
            continue
        (
            pixel_values,
            depth,
            valid_mask,
            condition,
            _,
        ) = unpack_ati_batch(batch, device)

        target_size = depth.shape[-2:]

        optimizer.zero_grad(set_to_none=True)

        prefix_head = "metric" if model_id.startswith("metric") else "relative"
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(
                pixel_values,
                condition,
                target_size=target_size,
            )
            if prefix_head == "relative":
                aligned = align_relative_prediction_to_depth_space(
                    out["base_depth"],
                    depth,
                    valid_mask,
                    align_mode=relative_align_mode,
                    sigma=out["std"],
                )
                aligned_mean = aligned["depth"] + out["camera_bias"]
                aligned_std = aligned["std"]
                aligned_log_var = torch.log(aligned_std.square() + 1e-8)
                # relative_uncertainty = aligned_std / ensure_bchw(aligned_mean).clamp_min(min_depth)
            else:
                aligned_mean = out["corrected_depth"]
                aligned_log_var = out["log_variance"]
                aligned_std = out["std"]
            t_mu_aligned = out["base_depth"] if prefix_head == "metric" else aligned["depth"]
            uncertainty_map = aligned_std
            
            nll_loss = gaussian_nll_depth_loss(
                aligned_mean,
                aligned_log_var,
                depth,
                valid_mask,
                lambda_smooth_logvar=lambda_smooth_logvar,
            )
            list_loss = image_level_listnet_loss(
                t_mu_aligned,
                uncertainty_map,
                depth,
                valid_mask,
                temperature=listnet_temperature,
                uncertainty_mode=uncertainty_mode,
            )
            loss = nll_loss + list_loss_weight * list_loss

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        mu_aligned = out["base_depth"].detach() if prefix_head == "metric" else aligned["depth"].detach()
        # std_aligned = aligned_std.detach()
        # relative_uncertainty = relative_uncertainty.detach()
        uncertainty_map = uncertainty_map.detach()
        
        batched_metrics = compute_comprehensive_depth_metrics(
            mu=mu_aligned,
            target=depth,
            valid_mask=valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        correlations = compute_loss_uncertainty_correlations(
            mu_aligned.detach(),
            depth,
            valid_mask,
            uncertainty=uncertainty_map.detach(),
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        ause_metrics = compute_sparsification_ause_metrics(
            mu_aligned,
            depth,
            valid_mask,
            uncertainty=uncertainty_map,
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        aurg_metrics = compute_sparsification_aurg_metrics(
            mu_aligned,
            depth,
            valid_mask,
            uncertainty=uncertainty_map,
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        
        running_loss += loss.item()
        running_nll_loss += nll_loss.item()
        running_list_loss += list_loss.item()
        running_abs_rel += batched_metrics["abs_rel"].mean().item()
        running_rmse += batched_metrics["rmse"].mean().item()
        running_a1 += batched_metrics["a1"].mean().item()
        running_ause_abs_rel.append(ause_metrics["ause_abs_rel"])
        running_aurg_abs_rel.append(aurg_metrics["aurg_abs_rel"])
        running_ause_a1.append(ause_metrics["ause_a1"])
        running_aurg_a1.append(aurg_metrics["aurg_a1"])
        running_pearson_correlation_l1.append(correlations["loss_uncertainty_pearson"])
        running_spearman_correlation_l1.append(correlations["loss_uncertainty_spearman"])
        processed_batches += 1

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{running_loss / step:.4f}",
            abs_rel=f"{running_abs_rel / step:.4f}",
            a1=f"{running_a1 / step:.4f}",
            ause_abs_rel=f"{torch.cat(running_ause_abs_rel, dim=0).mean().item():.4f}",
            ause_a1=f"{torch.cat(running_ause_a1, dim=0).mean().item():.4f}",
            correlation_l1=f"{torch.cat(running_pearson_correlation_l1, dim=0).mean().item():.4f}",
            spearman_correlation_l1=f"{torch.cat(running_spearman_correlation_l1, dim=0).mean().item():.4f}",
        )
        if log_interval > 0 and step % log_interval == 0:
            logger.info(
                "epoch=%d step=%d/%d avg_loss=%.6f abs_rel=%.6f a1=%.6f ause_abs_rel=%.6f aurg_abs_rel=%.6f ause_a1=%.6f aurg_a1=%.6f correlation_l1=%.6f",
                epoch,
                step,
                len(loader),
                running_loss / step,
                running_abs_rel / step,
                running_a1 / step,
                torch.cat(running_ause_abs_rel, dim=0).mean().item(),
                torch.cat(running_aurg_abs_rel, dim=0).mean().item(),
                torch.cat(running_ause_a1, dim=0).mean().item(),
                torch.cat(running_aurg_a1, dim=0).mean().item(),
                torch.cat(running_pearson_correlation_l1, dim=0).mean().item()
            )

        global_step += 1

    n = max(processed_batches, 1)
    epoch_metrics = {
        "loss": running_loss / n,
        "nll_loss": running_nll_loss / n,
        "list_loss": running_list_loss / n,
        "abs_rel": running_abs_rel / n,
        "rmse": running_rmse / n,
        "a1": running_a1 / n,
        "ause_abs_rel": torch.cat(running_ause_abs_rel, dim=0).mean().item(),
        "aurg_abs_rel": torch.cat(running_aurg_abs_rel, dim=0).mean().item(),
        "ause_a1": torch.cat(running_ause_a1, dim=0).mean().item(),
        "aurg_a1": torch.cat(running_aurg_a1, dim=0).mean().item(),
        "correlation_l1": torch.cat(running_pearson_correlation_l1, dim=0).mean().item(),
        "spearman_correlation_l1": torch.cat(running_spearman_correlation_l1, dim=0).mean().item(),
    }
    epoch_metrics.update({key: value / n for key, value in condition_sums.items()})
    epoch_metrics.update(_mean_finite_metrics(corr_sums, corr_counts))
    
    return epoch_metrics, global_step
