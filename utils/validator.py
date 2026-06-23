import torch
from torch.utils.data import DataLoader, Subset
from utils.train_utils import *
from model.loss_fn import gaussian_nll_depth_loss, image_level_listnet_loss
from evaluation_utils.eval_utils import (
    compute_comprehensive_depth_metrics,
)
from evaluation_utils.correlation_utils import (
    compute_loss_uncertainty_correlations,
    compute_sparsification_ause_metrics,
    compute_masked_correlations
)
from tqdm.auto import tqdm

def _metric_values_to_tensor(value):
    if torch.is_tensor(value):
        return value.detach().float().flatten().cpu()
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)], dtype=torch.float32)
    return None

def _finite_mean(value):
    values = _metric_values_to_tensor(value)
    finite_mask = torch.isfinite(values)
    if not finite_mask.any():
        return float("nan")
    return float(values[finite_mask].mean().item())


def _extend_metric_values(accumulator, metrics):
    for key, value in metrics.items():
        if not isinstance(accumulator.get(key), list):
            continue
        values = _metric_values_to_tensor(value)
        accumulator[key].append(values)

def _accumulate_validation_result(accumulator, result):
    if result is None:
        return

    accumulator["running_loss"] += result["loss"]
    accumulator["running_abs_rel"] += _finite_mean(result["batched_metrics"].get("abs_rel"))
    accumulator["running_rmse"] += _finite_mean(result["batched_metrics"].get("rmse"))
    accumulator["running_a1"] += _finite_mean(result["batched_metrics"].get("a1"))
    accumulator["running_corr_samples"] += result["correlations"].get("loss_uncertainty_samples", 0)
    accumulator["running_ause_samples"] += result["ause_metrics"].get("ause_samples", 0)
    accumulator["processed_batches"] += 1
    _extend_metric_values(accumulator, result["batched_metrics"])
    _extend_metric_values(accumulator, result["correlations"])
    _extend_metric_values(accumulator, result["ause_metrics"])


def _finalize_validation_accumulator(accumulator):
    val_metrics = {}
    for key, values in accumulator.items():
        if not isinstance(values, list):
            continue
        stacked_values = torch.cat(values, dim=0)
        finite_mask = torch.isfinite(stacked_values)
        val_metrics[key] = float(stacked_values[finite_mask].mean().item())
    return val_metrics


@torch.no_grad()
def validate(
    epoch: int,
    model_id: str,
    model,
    loader: DataLoader,
    device,
    amp: bool,
    lambda_smooth_logvar: float,
    listnet_temperature: float,
    uncertainty_mode: str,
    list_loss_weight: float,
    correlation_max_samples: int = 100_000,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    relative_align_mode: str = "scale_shift",
):
    model.eval()

    total_accumulator = {
        "running_loss": 0.0,
        "running_abs_rel": 0.0,
        "running_rmse": 0.0,
        "running_a1": 0.0,
        "running_corr_samples": 0,
        "running_ause_samples": 0,
        "processed_batches": 0,
        "abs_rel": [],
        "rmse": [],
        "a1": [],
        "uncertainty_mean": [],
        "ause_abs_rel": [],
        "ause_a1": [],
        "loss_uncertainty_pearson": [],
        "loss_uncertainty_spearman": [],
    }

    progress_bar = tqdm(
        loader,
        desc=f"Validation {epoch:03d}",
        dynamic_ncols=True,
        leave=False,
    )
    
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

        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(
                pixel_values,
                condition=condition,
                target_size=target_size,
            )
            nll_loss = gaussian_nll_depth_loss(
                out["mu"],
                out["log_var"],
                depth,
                valid_mask,
                lambda_smooth_logvar=lambda_smooth_logvar,
            )
            list_loss = image_level_listnet_loss(
                out["mu"],
                out["std"],
                depth,
                valid_mask,
                temperature=listnet_temperature,
                uncertainty_mode=uncertainty_mode,
            )
            loss = nll_loss + list_loss_weight * list_loss

        prefix_head = "metric" if model_id.startswith("metric") else "relative"
        batched_metrics = compute_comprehensive_depth_metrics(
            mu=out["mu"].detach(),
            target=depth,
            valid_mask=valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
            align_mode=relative_align_mode,
            depth_model_type=prefix_head,
        )
        correlations = compute_loss_uncertainty_correlations(
            out["mu"].detach(),
            out["log_var"].detach(),
            depth,
            valid_mask,
            uncertainty=out["std"].detach(),
            max_samples=correlation_max_samples,
        )
        ause_metrics = compute_sparsification_ause_metrics(
            out["mu"].detach(),
            depth,
            valid_mask,
            uncertainty=out["std"].detach(),
            max_samples=correlation_max_samples,
        )
        batch_uncertainty_mean = torch.mean(out["std"].detach(), dim=[1, 2, 3]) # [B]
        batch_result = {
            "loss": loss.item(),
            "prefix_head": prefix_head,
            "batched_metrics": batched_metrics,
            "correlations": correlations,
            "ause_metrics": ause_metrics,
            "uncertainty_mean": batch_uncertainty_mean
        }
        _accumulate_validation_result(total_accumulator, batch_result)
        
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_accumulator['running_loss'] / step:.4f}",
            abs_rel=f"{batched_metrics['abs_rel'].mean().item():.4f}",
            a1=f"{batched_metrics['a1'].mean().item():.4f}",
            ause_a1=f"{ause_metrics['ause_a1'].mean().item():.4f}",
            ause_abs_rel=f"{ause_metrics['ause_abs_rel'].mean().item():.4f}",
        )
    
    total_abs_rel_tensor = torch.cat(total_accumulator["abs_rel"], dim=0)
    total_a1_tensor = torch.cat(total_accumulator["a1"], dim=0)
    total_uncertainty_mean_tensor = torch.cat(total_accumulator["uncertainty_mean"], dim=0)
    
    a1_uncertainty_correlation = compute_masked_correlations(
        total_a1_tensor,
        total_uncertainty_mean_tensor,
        mask=torch.isfinite(total_a1_tensor) & torch.isfinite(total_uncertainty_mean_tensor),
        prefix="aggregated_a1_unc"
    )
    abs_rel_uncertainty_correlation = compute_masked_correlations(
        total_abs_rel_tensor,
        total_uncertainty_mean_tensor,
        mask=torch.isfinite(total_abs_rel_tensor) & torch.isfinite(total_uncertainty_mean_tensor),
        prefix="aggregated_abs_rel_unc"
    )
    
    return {
        **a1_uncertainty_correlation,
        **abs_rel_uncertainty_correlation,
        **_finalize_validation_accumulator(total_accumulator)
    }
