from typing import Tuple, Dict
import torch
from dataset.ati_dataset_caminduce import flatten_group_batch
from model.loss_fn import (
    fheteroscedastic_caminduced_depth_loss, 
    gap_weighted_ranknet_loss,
    camera_risk_scores
) # , image_level_listnet_loss
from evaluation_utils.eval_utils import (
    align_relative_prediction_to_depth_space,
    _mean_finite_metrics,
)
from evaluation_utils.eval_metrics import (
    compute_comprehensive_depth_metrics,
    compute_loss_uncertainty_correlations,
    compute_sparsification_ause_metrics,
    compute_vector_masked_correlations,
    compute_camera_induced_degradation_values,
    summarize_camera_induced_degradation_correlations
)
from tqdm.auto import tqdm
from utils.train_utils import *
import logging

def get_batched_correlations(batched_metric, uncertainty, max_samples=100_000):
    abs_rel = torch.cat(batched_metric["abs_rel"], dim=0)
    a1 = torch.cat(batched_metric["a1"], dim=0)
    uncertainty_mean = torch.cat(uncertainty, dim=0)
    a1_uncertainty_correlation = compute_vector_masked_correlations( 
        a1,
        uncertainty_mean,
        valid_mask=torch.isfinite(a1) & torch.isfinite(uncertainty_mean),
        prefix="aggregated_a1_unc",
        max_samples=max_samples
    )
    abs_rel_uncertainty_correlation = compute_vector_masked_correlations(
        abs_rel,
        uncertainty_mean,
        valid_mask=torch.isfinite(abs_rel) & torch.isfinite(uncertainty_mean),
        prefix="aggregated_abs_rel_unc",
        max_samples=max_samples
    )
    return a1_uncertainty_correlation, abs_rel_uncertainty_correlation


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
    lambda_variance: float,
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
    running_mean_loss = 0.0
    running_variance_loss = 0.0
    running_list_loss = 0.0
    running_abs_rel = 0.0
    running_rmse = 0.0
    running_a1 = 0.0
    running_ause_abs_rel = []
    running_ause_a1 = []
    running_pearson_correlation_l1 = []
    running_spearman_correlation_l1 = []
    
    running_batched_metrics = {
        "abs_rel": [],
        "a1": [],
    }
    running_uncertainty_mean = []
    running_degradation_values = {
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
    corr_sums = {}
    corr_counts = {}
    condition_sums = {}
    processed_batches = 0

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
        group_ids = batch["group_index"].to(device=device)[:, None].expand(-1, num_candidates).reshape(-1)
        
        target_size = candidate_depth.shape[-2:]
        optimizer.zero_grad(set_to_none=True)
        
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
                raise NotImplementedError("Metric head is not implemented in this training function.")
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

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        mu_aligned = aligned["depth"].detach()
        uncertainty_map = uncertainty_map.detach()
        batched_metrics = compute_comprehensive_depth_metrics(
            mu=mu_aligned,
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
            mu_aligned,
            candidate_depth,
            candidate_valid_mask,
            uncertainty=uncertainty_map,
            max_samples=correlation_max_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        
        running_loss += loss.item()
        running_nll_loss += nll_loss.item()
        running_mean_loss += mean_loss.item()
        running_variance_loss += variance_loss.item()
        running_list_loss += ranking_loss.item()
        running_abs_rel += batched_metrics["abs_rel"].mean().item()
        running_rmse += batched_metrics["rmse"].mean().item()
        running_a1 += batched_metrics["a1"].mean().item()
        running_ause_abs_rel.append(ause_metrics["ause_abs_rel"])
        running_ause_a1.append(ause_metrics["ause_a1"])
        running_pearson_correlation_l1.append(correlations["loss_uncertainty_pearson"])
        running_spearman_correlation_l1.append(correlations["loss_uncertainty_spearman"])
        
        running_batched_metrics["abs_rel"].append(batched_metrics["abs_rel"])
        running_batched_metrics["a1"].append(batched_metrics["a1"])
        running_uncertainty_mean.append(masked_image_mean(uncertainty_map, candidate_valid_mask))
        degradation_values = compute_camera_induced_degradation_values(
            candidate_metrics=batched_metrics,
            canonical_metrics=canonical_batched_metrics,
            camera_bias=out["camera_bias"],
            variance=out["variance"],
            valid_mask=candidate_valid_mask,
            group_ids=group_ids,
        )
        for key, value in degradation_values.items():
            running_degradation_values[key].append(value.detach().cpu())
        running_a1_unc_corr, running_abs_rel_unc_corr = get_batched_correlations(
            running_batched_metrics,
            running_uncertainty_mean,
            max_samples=correlation_max_samples,
        )
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
                "epoch=%d step=%d/%d avg_loss=%.6f abs_rel=%.6f a1=%.6f sample_a1_corr=%.6f sample_abs_rel_corr=%.6f ause_abs_rel=%.6f ause_a1=%.6f correlation_l1=%.6f",
                epoch,
                step,
                len(loader),
                running_loss / step,
                running_abs_rel / step,
                running_a1 / step,
                running_a1_unc_corr["aggregated_a1_unc_pearson"],
                running_abs_rel_unc_corr["aggregated_abs_rel_unc_pearson"],
                torch.cat(running_ause_abs_rel, dim=0).mean().item(),
                torch.cat(running_ause_a1, dim=0).mean().item(),
                torch.cat(running_pearson_correlation_l1, dim=0).mean().item()
            )
        global_step += 1

    total_a1_unc_corr, total_abs_rel_unc_corr = get_batched_correlations(
        running_batched_metrics,
        running_uncertainty_mean,
        max_samples=correlation_max_samples,
    )
    degradation_metrics = summarize_camera_induced_degradation_correlations(
        {
            key: torch.cat(values, dim=0)
            for key, values in running_degradation_values.items()
            if values
        },
        max_samples=correlation_max_samples,
    )

    n = max(processed_batches, 1)
    epoch_metrics = {
        "loss": running_loss / n,
        "nll_loss": running_nll_loss / n,
        "mean_loss": running_mean_loss / n,
        "variance_loss": running_variance_loss / n,
        "list_loss": running_list_loss / n,
        "abs_rel": running_abs_rel / n,
        "rmse": running_rmse / n,
        "a1": running_a1 / n,
        "sample_a1_pearson": total_a1_unc_corr["aggregated_a1_unc_pearson"],
        "sample_abs_rel_pearson": total_abs_rel_unc_corr["aggregated_abs_rel_unc_pearson"],
        "ause_abs_rel": torch.cat(running_ause_abs_rel, dim=0).mean().item(),
        "ause_a1": torch.cat(running_ause_a1, dim=0).mean().item(),
        "correlation_l1": torch.cat(running_pearson_correlation_l1, dim=0).mean().item(),
        "spearman_correlation_l1": torch.cat(running_spearman_correlation_l1, dim=0).mean().item(),
    }
    epoch_metrics.update({key: value / n for key, value in condition_sums.items()})
    epoch_metrics.update(_mean_finite_metrics(corr_sums, corr_counts))
    epoch_metrics.update(degradation_metrics)
    
    return epoch_metrics, global_step
