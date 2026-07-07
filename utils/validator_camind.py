import torch
from torch.utils.data import DataLoader, Subset
from dataset.ati_dataset_caminduce import flatten_group_batch
from utils.train_utils import *
from model.loss_fn import (
    fheteroscedastic_caminduced_depth_loss, 
    gap_weighted_ranknet_loss,
    camera_risk_scores
)
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
    _extend_metric_values(accumulator, result["degradation_values"])


def _finalize_validation_accumulator(accumulator):
    val_metrics = {}
    for key, values in accumulator.items():
        if not isinstance(values, list):
            continue
        if key == "group_id":
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
    degradation_keys = (
        "B2",
        "V",
        "R",
        "sqrt_R",
        "log_R",
        "abs_rel_degradation",
        "delta1_degradation",
        "delta1_error_degradation",
        "group_id",
    )
    if accumulator.get("R") and accumulator.get("group_id"):
        degradation_values = {
            key: torch.cat(accumulator[key], dim=0)
            for key in degradation_keys
            if accumulator.get(key)
        }
        val_metrics.update(summarize_camera_induced_degradation_correlations(degradation_values))
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
        "degradation_values": _select_metric_values(result["degradation_values"], sample_mask),
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
        "rmsu": [],
        "B2": [],
        "V": [],
        "R": [],
        "sqrt_R": [],
        "log_R": [],
        "abs_rel_degradation": [],
        "delta1_degradation": [],
        "delta1_error_degradation": [],
        "group_id": [],
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
    lambda_variance: float,
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
        num_groups, num_candidates = (
            batch["candidate_images"].shape[:2]
        )
        flat_batch = tensor_device(flatten_group_batch(batch), device)
        candidate_imgs = flat_batch["candidate_images"]
        canonical_imgs = flat_batch["canonical_images"]
        candidate_depth = flat_batch["candidate_depths"]
        canonical_depth = flat_batch["canonical_depths"]
        candidate_valid_mask = flat_batch["candidate_valid_mask"]
        canonical_valid_mask = torch.isfinite(canonical_depth)
        canonical_valid_mask &= canonical_depth > min_depth
        canonical_valid_mask &= canonical_depth < max_depth
        candidate_condition = flat_batch["camera_context"]

        target_size = candidate_depth.shape[-2:]
        group_ids = batch["group_index"].to(device=device)[:, None].expand(-1, num_candidates).reshape(-1)
        topology_number = batch["info"][:, 6].to(device=device).long() # The topology number is stored in the 7th column of the info tensor
        topology_number = topology_number[:, None].expand(-1, num_candidates)
        topology_number = topology_number.reshape(-1)      # [G*K]

        prefix_head = "metric" if model_id.startswith("metric") else "relative"
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(
                candidate_imgs,
                canonical_imgs,
                candidate_condition,
                target_size=target_size,
            )
            if prefix_head == "relative":
                aligned = align_relative_prediction_to_depth_space(
                    out["candidate_depth"],
                    candidate_depth,
                    candidate_valid_mask,
                    align_mode=relative_align_mode,
                )
                canonical_aligned = align_relative_prediction_to_depth_space(
                    out["canonical_depth"],
                    canonical_depth,
                    canonical_valid_mask,
                    align_mode=relative_align_mode,
                )
                aligned_std = out["std"]
            else:
                raise NotImplementedError("Metric model validation is not implemented yet.")
            uncertainty_map = torch.sqrt(out["camera_bias"].square() + aligned_std.square())
            
            mean_loss, variance_loss = fheteroscedastic_caminduced_depth_loss(
                out["corrected_depth"],
                out["variance"],
                out["canonical_depth"],
                lambda_smooth_logvar=lambda_smooth_logvar,
            )
            group_canonical_depth = reshape_group_batch(out["canonical_depth"], num_groups, num_candidates)
            group_candidate_depth = reshape_group_batch(out["candidate_depth"], num_groups, num_candidates)
            cam_bias = reshape_group_batch(out["camera_bias"], num_groups, num_candidates)
            cam_variance = reshape_group_batch(out["raw_variance"], num_groups, num_candidates)
            predicted_risk, target_risk = camera_risk_scores(
                bias=cam_bias,
                variance=cam_variance,
                canonical_delta=group_canonical_depth - group_candidate_depth,
                valid_mask=torch.ones_like(
                    group_candidate_depth, dtype=torch.float32
                ).to(device),
            )
            ranking_loss = gap_weighted_ranknet_loss(
                predicted_risk,
                target_risk, 
                temperature=listnet_temperature,
                eps=1e-6
            )
            nll_loss = mean_loss + lambda_variance * variance_loss
            loss = nll_loss + list_loss_weight * ranking_loss

        mu_aligned = aligned["depth"].detach()
        std_aligned = aligned_std.detach()
        uncertainty_map = uncertainty_map.detach()

        batched_metrics = compute_comprehensive_depth_metrics(
            mu=mu_aligned.detach(),
            target=candidate_depth,
            valid_mask=candidate_valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        canonical_batched_metrics = compute_comprehensive_depth_metrics(
            mu=canonical_aligned["depth"].detach(),
            target=canonical_depth,
            valid_mask=canonical_valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        correlations = compute_loss_uncertainty_correlations(
            mu_aligned.detach(),
            candidate_depth,
            candidate_valid_mask,
            uncertainty=uncertainty_map.detach(),
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        ause_metrics = compute_sparsification_ause_metrics(
            mu_aligned.detach(),
            candidate_depth,
            candidate_valid_mask,
            uncertainty=uncertainty_map.detach(),
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        aurg_metrics = compute_sparsification_aurg_metrics(
            mu_aligned.detach(),
            candidate_depth,
            candidate_valid_mask,
            uncertainty=uncertainty_map.detach(),
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        aru_rmsu_metrics = compute_aru_rmsu_metrics(
            mu_aligned.detach(),
            candidate_depth,
            candidate_valid_mask,
            uncertainty=uncertainty_map.detach(),
            min_depth=min_depth,
            max_depth=max_depth,
        )
        batch_uncertainty_mean = masked_image_mean(uncertainty_map, candidate_valid_mask)
        batch_depth_uncertainty_mean = masked_image_mean(std_aligned, candidate_valid_mask)
        degradation_values = compute_camera_induced_degradation_values(
            candidate_metrics=batched_metrics,
            canonical_metrics=canonical_batched_metrics,
            camera_bias=out["camera_bias"],
            variance=out["variance"],
            valid_mask=candidate_valid_mask,
            group_ids=group_ids,
        )
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
            },
            "degradation_values": degradation_values,
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
