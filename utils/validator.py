import torch
from torch.utils.data import DataLoader, Subset
from utils.train_utils import *
from model.loss_fn import gaussian_nll_depth_loss, image_level_listnet_loss
from evaluation_utils.eval_metrics import *
from evaluation_utils.eval_utils import (
    align_relative_prediction_to_depth_space,
    ensure_bchw,
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


def _masked_image_mean(values, valid_mask):
    mask = valid_mask.bool()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    counts = mask.flatten(1).sum(dim=1)
    means = (
        torch.where(mask, values, torch.zeros_like(values))
        .flatten(1)
        .sum(dim=1)
        / counts.clamp_min(1).to(dtype=values.dtype)
    )
    return torch.where(
        counts > 0,
        means,
        means.new_full(means.shape, float("nan")),
    )


def _extend_metric_values(accumulator, metrics):
    for key, value in metrics.items():
        if not isinstance(accumulator.get(key), list):
            continue
        values = _metric_values_to_tensor(value)
        if values is None:
            continue
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
    _extend_metric_values(accumulator, result["uncertainty_mean"])
    _extend_metric_values(accumulator, result["correlations"])
    _extend_metric_values(accumulator, result["ause_metrics"])
    _extend_metric_values(accumulator, result["aurg_metrics"])
    _extend_metric_values(accumulator, result["aru_rmsu"])


def _finalize_validation_accumulator(accumulator):
    val_metrics = {}
    for key, values in accumulator.items():
        if not isinstance(values, list):
            continue
        if not values:
            continue
        stacked_values = torch.cat(values, dim=0)
        finite_mask = torch.isfinite(stacked_values)
        val_metrics[key] = (
            float(stacked_values[finite_mask].mean().item())
            if finite_mask.any()
            else float("nan")
        )

    if accumulator.get("abs_rel") and accumulator.get("uncertainty_mean"):
        abs_rel = torch.cat(accumulator["abs_rel"], dim=0)
        a1 = torch.cat(accumulator["a1"], dim=0)
        uncertainty_mean = torch.cat(accumulator["uncertainty_mean"], dim=0)
        a1_uncertainty_correlation = compute_vector_masked_correlations( 
            a1,
            uncertainty_mean,
            valid_mask=torch.isfinite(a1) & torch.isfinite(uncertainty_mean),
            prefix="aggregated_a1_unc"
        )
        abs_rel_uncertainty_correlation = compute_vector_masked_correlations(
            abs_rel,
            uncertainty_mean,
            valid_mask=torch.isfinite(abs_rel) & torch.isfinite(uncertainty_mean),
            prefix="aggregated_abs_rel_unc"
        )
        val_metrics.update(a1_uncertainty_correlation)
        val_metrics.update(abs_rel_uncertainty_correlation)
    return val_metrics


def _select_metric_values(metrics, sample_mask):
    return {
        key: value[sample_mask]
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == sample_mask.shape[0]
        else value
        for key, value in metrics.items()
    }


def _select_batch_result(result, sample_mask):
    if not sample_mask.any().item():
        return None

    return {
        "loss": result["loss"],
        "prefix_head": result["prefix_head"],
        "batched_metrics": _select_metric_values(result["batched_metrics"], sample_mask),
        "ause_metrics": _select_metric_values(result["ause_metrics"], sample_mask),
        "aurg_metrics": _select_metric_values(result["aurg_metrics"], sample_mask),
        "correlations": _select_metric_values(result["correlations"], sample_mask),
        "aru_rmsu": _select_metric_values(result["aru_rmsu"], sample_mask),
        "uncertainty_mean": _select_metric_values(result["uncertainty_mean"], sample_mask),
    }

def __create_accumulator():
    return {
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
        "loss_uncertainty_pearson": [],
        "loss_uncertainty_spearman": [],
        "uncertainty_mean": [],
        "depth_uncertainty_mean": [],
        "ause_abs_rel": [],
        "ause_a1": [],
        "aurg_abs_rel": [],
        "aurg_a1": [],
        "aru": [],
        "rmsu": []
    }

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
    seen_topology_numbers: torch.Tensor = None,
    unseen_topology_numbers: torch.Tensor = None,
    correlation_max_samples: int = 100_000,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    relative_align_mode: str = "scale_shift",
):
    model.eval()

    total_accumulator = __create_accumulator()
    seen_accumulator = __create_accumulator()
    unseen_accumulator = __create_accumulator()

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
            info,
        ) = unpack_ati_batch(batch, device)

        target_size = depth.shape[-2:]
        topology_number = info[:, 6].to(device=device).long() # The topology number is stored in the 7th column of the info tensor

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
                aligned_log_var = out["log_variance"]
                aligned_std = aligned["std"]
                # relative_uncertainty = aligned_std / ensure_bchw(aligned_mean).clamp_min(min_depth)
            else:
                aligned_mean = out["corrected_depth"]
                aligned_log_var = out["log_variance"]
                aligned_std = out["std"]
            t_mu_aligned = out["base_depth"] if prefix_head == "metric" else aligned["depth"]
            uncertainty_map = aligned_std # if prefix_head == "metric" else relative_uncertainty
            
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

        mu_aligned = out["base_depth"].detach() if prefix_head == "metric" else aligned["depth"].detach()
        std_aligned = aligned_std.detach()
        uncertainty_map = uncertainty_map.detach()

        batched_metrics = compute_comprehensive_depth_metrics(
            mu=mu_aligned.detach(),
            target=depth,
            valid_mask=valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        correlations = compute_loss_uncertainty_correlations(
            mu_aligned.detach(),
            depth,
            valid_mask,
            uncertainty=std_aligned.detach(),
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        ause_metrics = compute_sparsification_ause_metrics(
            mu_aligned.detach(),
            depth,
            valid_mask,
            uncertainty=uncertainty_map.detach(),
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        aurg_metrics = compute_sparsification_aurg_metrics(
            mu_aligned.detach(),
            depth,
            valid_mask,
            uncertainty=uncertainty_map.detach(),
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        aru_rmsu_metrics = compute_aru_rmsu_metrics(
            mu_aligned.detach(),
            depth,
            valid_mask,
            uncertainty=std_aligned.detach(),
            min_depth=min_depth,
            max_depth=max_depth,
        )
        batch_uncertainty_mean = _masked_image_mean(uncertainty_map, valid_mask)
        batch_depth_uncertainty_mean = _masked_image_mean(std_aligned, valid_mask)
        batch_result = {
            "loss": loss.item(),
            "prefix_head": prefix_head,
            "batched_metrics": batched_metrics,
            "ause_metrics": ause_metrics,
            "aurg_metrics": aurg_metrics,
            "correlations": correlations,
            "aru_rmsu": aru_rmsu_metrics,
            "uncertainty_mean": {
                "uncertainty_mean": batch_uncertainty_mean,
                "depth_uncertainty_mean": batch_depth_uncertainty_mean,
            }
        }
        _accumulate_validation_result(total_accumulator, batch_result)
        if seen_topology_numbers is not None:
            seen_mask = torch.isin(topology_number, seen_topology_numbers.to(device=device).long())
            _accumulate_validation_result(
                seen_accumulator,
                _select_batch_result(batch_result, seen_mask),
            )
        if unseen_topology_numbers is not None:
            unseen_mask = torch.isin(topology_number, unseen_topology_numbers.to(device=device).long())
            _accumulate_validation_result(
                unseen_accumulator,
                _select_batch_result(batch_result, unseen_mask),
            )
        
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_accumulator['running_loss'] / step:.4f}",
            abs_rel=f"{batched_metrics['abs_rel'].mean().item():.4f}",
            a1=f"{batched_metrics['a1'].mean().item():.4f}",
            ause_a1=f"{ause_metrics['ause_a1'].mean().item():.4f}",
            ause_abs_rel=f"{ause_metrics['ause_abs_rel'].mean().item():.4f}",
        )
    
    total_metrics = _finalize_validation_accumulator(total_accumulator)
    seen_metrics = _finalize_validation_accumulator(seen_accumulator)
    unseen_metrics = _finalize_validation_accumulator(unseen_accumulator)

    return total_metrics, seen_metrics, unseen_metrics

# compute_comprehensive_depth_metrics,
# from evaluation_utils.correlation_utils import (
#     compute_loss_uncertainty_correlations,
#     compute_sparsification_ause_metrics,
#     compute_sparsification_aurg_metrics,
#     compute_masked_correlations,
#     compute_aru_rmsu_metrics
# )
