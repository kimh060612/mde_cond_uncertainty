import math
import torch
from torch.utils.data import DataLoader, Subset
import hydra
import logging
from glob import glob
from pathlib import Path
from transformers import AutoImageProcessor
from dataset.ati_dataset_caminduce import (
    CameraParameterRange,
    PairedResizeToTensor,
    LIGHT_LEVELS,
    MOTION_LEVELS,
    FoundationCameraGroupedDataset
)
from dataset.ati_dataset_caminduce import *
from evaluation_utils.eval_selection import plot_selection_alpha_sweep
from model.dav2_ati_model import MODEL_IDS
from model.dav2_camerror_model import CameraInducedErrorModel
from omegaconf import DictConfig, OmegaConf
from utils.train_utils import *
from utils.trainer_camind import train_one_epoch
from utils.validator_camind import validate
from utils.logger import setup_logger

from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None


@hydra.main(config_path="config", config_name="base_caminduce")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") and (not cfg.training.no_amp)
    wandb_run = None
    model_id = MODEL_IDS[cfg.model.model_id]
    print(f"Using model: {model_id}")
    print(f"dataset root: {cfg.dataset.dataset_root}")
    print(f"wandb project: {cfg.training.wandb_entity}/{cfg.training.wandb_project}")
    
    seed = cfg.training.seed
    seed_everything(seed)
    
    if not cfg.training.disable_wandb:
        if wandb is None:
            raise ImportError(
                "wandb is required for logging. Install it with `pip install wandb` "
                "or pass `--disable_wandb` for a local run without wandb."
            )
        wandb_run = wandb.init(
            entity=cfg.training.wandb_entity,
            project=cfg.training.wandb_project,
            name=cfg.training.wandb_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        
    image_processor = AutoImageProcessor.from_pretrained(
        model_id,
        cache_dir=None,
        use_fast=False,
    )
    
    seen_val_topologies = [ str(topology).strip() for topology in cfg.dataset.seen_val_topologies ]
    unseen_val_topologies = [ str(topology).strip() for topology in cfg.dataset.unseen_val_topologies ]
    seen_val_topology_ids = {topology_id(topology) for topology in seen_val_topologies}
    unseen_val_topology_ids = {topology_id(topology) for topology in unseen_val_topologies}
    seen_topology_idx = torch.Tensor(list(seen_val_topology_ids)).long()
    unseen_topology_idx = torch.Tensor(list(unseen_val_topology_ids)).long()
    print("[Training] Seen validation topologies:", seen_topology_idx)
    print("[Training] Unseen validation topologies:", unseen_topology_idx)

    csv_root = Path(cfg.dataset.csv_path)
    if csv_root.is_file():
        csv_paths = [str(csv_root)]
    else:
        csv_paths = sorted(glob(f"{cfg.dataset.csv_path}/*.csv"))
    if len(csv_paths) == 0:
        raise ValueError(f"No CSV files found in {cfg.dataset.csv_path}")
        
    train_set = FoundationCameraGroupedDataset(
        csv_paths=csv_paths,
        foundation_model_name=cfg.model.model_id,
        camera_model_name=cfg.model.camera_model_name,
        parameter_range=CameraParameterRange(
            exposure_min=cfg.dataset.exposure_min,
            exposure_max=cfg.dataset.exposure_max,
            gain_min=cfg.dataset.gain_min,
            gain_max=cfg.dataset.gain_max,
        ),
        candidates_per_group=cfg.training.candidates_per_group,
        candidate_sampling="parameter_diverse",
        parameter_normalization="linear",
        context_output_range="zero_one",
        path_replacements={
            "/media/michael/ssd1/AIoT_ATI/orbbec_realworld_dataset":
            cfg.dataset.dataset_root,
        },
        pair_transform=PairedResizeToTensor(image_processor=image_processor),
        min_overlap_ratio=cfg.dataset.min_registration_overlap_ratio,
        min_ecc_score=cfg.dataset.min_registration_ecc_score,
        max_time_diff_sec=cfg.dataset.max_pair_time_diff_sec,
        max_registration_translation_px=cfg.dataset.max_registration_translation_px,
        abs_rel_degradation_quantile=cfg.dataset.abs_rel_degradation_quantile,
        topologies=cfg.dataset.train_topologies,
        load_images=True,
        load_depth=False,
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        seed=cfg.training.seed,
    )
    val_set = FoundationCameraGroupedDataset(
        csv_paths=csv_paths,
        foundation_model_name=cfg.model.model_id,
        camera_model_name=cfg.model.camera_model_name,
        parameter_range=train_set.parameter_range,
        candidates_per_group=cfg.evaluation.min_camera_settings,
        candidate_sampling="parameter_diverse",
        parameter_normalization="linear",
        context_output_range="zero_one",
        path_replacements={
            "/media/michael/ssd1/AIoT_ATI/orbbec_realworld_dataset":
            cfg.dataset.dataset_root,
        },
        pair_transform=PairedResizeToTensor(image_processor=image_processor),
        min_overlap_ratio=cfg.dataset.min_registration_overlap_ratio,
        min_ecc_score=cfg.dataset.min_registration_ecc_score,
        max_time_diff_sec=cfg.dataset.max_pair_time_diff_sec,
        max_registration_translation_px=cfg.dataset.max_registration_translation_px,
        abs_rel_degradation_quantile=None,
        include_canonical_setting_as_candidate=True,
        use_all_candidates=True,
        topologies=list(cfg.dataset.seen_val_topologies) + list(cfg.dataset.unseen_val_topologies),
        load_images=True,
        load_depth=False,
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        seed=cfg.training.seed,
    )
    # copy_condition_normalization(val_set, train_set)
    # validation_topology_counts = count_items_by_topology(val_set)
    # seen_val_indices = topology_subset_indices(val_set, seen_val_topology_ids)
    # unseen_val_indices = topology_subset_indices(val_set, unseen_val_topology_ids)
    # seen_val_count = len(seen_val_indices)
    # unseen_val_count = len(unseen_val_indices)
    
    dataset_metadata = {
        "condition_names": list(train_set.camera_context_names),
        "light_levels": list(train_set.light_levels),
        "speed_levels": list(train_set.motion_levels),
        "exposure_min": train_set.parameter_range.exposure_min,
        "exposure_max": train_set.parameter_range.exposure_max,
        "gain_min": train_set.parameter_range.gain_min,
        "gain_max": train_set.parameter_range.gain_max,
        "min_valid_depth_ratio": cfg.dataset.min_valid_depth_ratio,
        # "train_scan_stats": dict(train_set.scan_stats),
        # "validation_scan_stats": dict(val_set.scan_stats),
        # "validation_topology_counts": dict(validation_topology_counts),
        "seen_validation_topologies": list(seen_val_topologies),
        # "seen_validation_samples": seen_val_count,
        "unseen_validation_topologies": list(unseen_val_topologies),
        # "unseen_validation_samples": unseen_val_count,
    }
    print(
        f"train samples: {len(train_set):,}, "
        f"val samples: {len(val_set):,}, "
        # f"seen val samples: {seen_val_count:,}, "
        # f"unseen val samples: {unseen_val_count:,}"
    )
    print(f"condition dim: {train_set.condition_dim}, names={list(train_set.camera_context_names)}")
    # print(f"train scan stats: {train_set.scan_stats}")
    # print(f"validation scan stats: {val_set.scan_stats}")
    # print(f"validation topology counts: {validation_topology_counts}")
    
    if wandb_run is not None:
        wandb_run.config.update(dataset_metadata, allow_val_change=True)
    
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=pin_memory,
    )

    model = CameraInducedErrorModel(
        model_id=model_id,
        context_dim=train_set.condition_dim,
        cache_dir=None,
        feature_channels=cfg.model.uncertainty_width,
        hidden_channels=cfg.model.uncertainty_width,
        film_hidden_dim=cfg.model.film_layer_width,
        max_bias=cfg.training.max_bias,
        min_log_variance=cfg.training.min_log_var,
        max_log_variance=cfg.training.max_log_var,
        initial_std=cfg.training.initial_std,
        variance_head_init_std=cfg.training.variance_head_init_std,
    ).to(device)
    
    backbone_params = []
    uncertainty_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("depth_model."):
            backbone_params.append(param)
        else:
            uncertainty_params.append(param)
    print(
        "trainable params: "
        f"backbone={sum(p.numel() for p in backbone_params):,}, "
        f"uncertainty={sum(p.numel() for p in uncertainty_params):,}"
    )
    
    param_groups = []
    if backbone_params:
        param_groups.append({"name": "backbone", "params": backbone_params, "lr": cfg.training.lr_backbone})
    if uncertainty_params:
        param_groups.append({"name": "uncertainty", "params": uncertainty_params, "lr": cfg.training.lr_uncertainty})
    if not param_groups:
        raise ValueError("No trainable parameters found")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.training.weight_decay)
    fallback_monitor = "q_vs_abs_rel_degradation_spearman"
    scheduler_monitor = cfg.training.get("lr_scheduler_monitor", fallback_monitor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.training.lr_scheduler_factor,
        patience=cfg.training.lr_scheduler_patience,
        threshold=cfg.training.lr_scheduler_threshold,
        threshold_mode="rel",
        cooldown=cfg.training.lr_scheduler_cooldown,
        min_lr=[
            group["lr"] * cfg.training.lr_scheduler_min_lr_ratio
            for group in optimizer.param_groups
        ],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_abs_rel_correlation = float("-inf")
    global_step = 0
    logger = setup_logger(
        name="depth_uncertainty_training",
        log_dir="./outputs/experiment_01/logs",
        level=logging.INFO,
    )
    logger.info("Training started")
    logger.info("Model: %s", model_id)
    
    for epoch in range(1, cfg.training.num_epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model_id=model_id,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp=amp,
            logger=logger,
            lambda_smooth_logvar=cfg.training.lambda_smooth_logvar,
            lambda_variance=cfg.training.lambda_variance,
            list_loss_weight=cfg.training.list_loss_weight,
            listnet_temperature=cfg.training.listnet_temperature,
            uncertainty_mode=cfg.training.uncertainty_mode,
            grad_clip=cfg.training.grad_clip,
            min_depth=cfg.dataset.min_depth,
            max_depth=cfg.dataset.max_depth,
            relative_align_mode=cfg.training.relative_align_mode,
            uncertainty_alpha=cfg.training.get("uncertainty_alpha", 1.0),
            global_step=global_step,
            log_interval=cfg.training.log_interval,
        )
        (
            val_total_metrics,
            val_seen_metrics,
            val_unseen_metrics,
            selection_sweeps,
        ) = validate(
            epoch=epoch,
            model_id=model_id,
            model=model,
            loader=val_loader,
            device=device,
            amp=amp,
            seen_topology_numbers=seen_topology_idx,
            unseen_topology_numbers=unseen_topology_idx,
            lambda_smooth_logvar=cfg.training.lambda_smooth_logvar,
            lambda_variance=cfg.training.lambda_variance,
            list_loss_weight=cfg.training.list_loss_weight,
            listnet_temperature=cfg.training.listnet_temperature,
            uncertainty_mode=cfg.training.uncertainty_mode,
            correlation_max_samples=cfg.training.correlation_max_samples,
            min_depth=cfg.dataset.min_depth,
            max_depth=cfg.dataset.max_depth,
            relative_align_mode=cfg.training.relative_align_mode,
            uncertainty_alpha=cfg.training.get("uncertainty_alpha", 1.0),
            selection_min_settings=cfg.evaluation.min_camera_settings,
            selection_thresholds=cfg.evaluation.relative_regret_thresholds_percent,
            selection_alpha_values=cfg.evaluation.alpha_sweep_values,
        )
        
        print(f"[epoch {epoch}] train={train_metrics}")
        print(f"[epoch {epoch}] val={val_total_metrics}")
        print(f"[epoch {epoch}] seen_val={val_seen_metrics}")
        print(f"[epoch {epoch}] unseen_val={val_unseen_metrics}")

        scheduler_metric = float(
            val_total_metrics.get(
                scheduler_monitor,
                val_total_metrics.get(fallback_monitor, float("nan")),
            )
        )
        if math.isfinite(scheduler_metric):
            scheduler.step(scheduler_metric)
        else:
            logger.warning(
                "Skipping LR scheduler step because %s is not finite: %s",
                scheduler_monitor,
                scheduler_metric,
            )
        
        best_metric = float(val_total_metrics.get(fallback_monitor, float("-inf")))
        is_best = best_metric > best_abs_rel_correlation
        if is_best:
            best_abs_rel_correlation = best_metric
            checkpoint_val_metrics = {
                **val_total_metrics,
                **prefix_metrics("val_seen", val_seen_metrics),
                **prefix_metrics("val_unseen", val_unseen_metrics),
            }
            save_checkpoint(
                model,
                image_processor,
                cfg.dataset.output_dir,
                epoch,
                checkpoint_val_metrics,
                dataset_metadata,
            )
            if wandb_run is not None:
                wandb_run.summary["best_q_abs_rel_degradation_spearman"] = best_abs_rel_correlation
                wandb_run.summary["best_epoch"] = epoch

        if wandb_run is not None:
            selection_figure = plot_selection_alpha_sweep(selection_sweeps)
            wandb_run.log({
                "epoch": epoch,
                "best/q_abs_rel_degradation_spearman": best_abs_rel_correlation,
                "best/is_best": int(is_best),
                "lr_scheduler/monitor": scheduler_metric,
                **{
                    f"lr/{group.get('name', group_idx)}": group["lr"]
                    for group_idx, group in enumerate(optimizer.param_groups)
                },
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{
                    f"val/{key}": value
                    for key, value in val_total_metrics.items()
                    if not key.startswith("selection_")
                },
                **{
                    f"val_seen/{key}": value
                    for key, value in val_seen_metrics.items()
                    if not key.startswith("selection_")
                },
                **{
                    f"val_unseen/{key}": value
                    for key, value in val_unseen_metrics.items()
                    if not key.startswith("selection_")
                },
                "val_seen/selection_accuracy": val_seen_metrics[
                    "selection_accuracy"
                ],
                "val_seen/selection_mean_regret_abs_rel": val_seen_metrics[
                    "selection_mean_regret_abs_rel"
                ],
                "val_unseen/selection_accuracy": val_unseen_metrics[
                    "selection_accuracy"
                ],
                "val_unseen/selection_mean_regret_abs_rel": val_unseen_metrics[
                    "selection_mean_regret_abs_rel"
                ],
                "val/selection_alpha_sweep": wandb.Image(selection_figure),
            }, step=epoch, commit=True)
            import matplotlib.pyplot as plt
            plt.close(selection_figure)
    
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
