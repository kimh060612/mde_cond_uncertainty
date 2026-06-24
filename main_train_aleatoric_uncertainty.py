import torch
from torch.utils.data import DataLoader, Subset
import wandb
from transformers import AutoImageProcessor
from dataset.ati_dataset_refactored import (
    ATIRealWorldUncertaintyDataset,
    ATIRealWorldUncertaintyValidationDataset,
    ati_collate_fn,
    LIGHT_LEVELS,
    MOTION_LEVELS,
)
from model.dav2_ati_model import ConditionedGaussianDepthAnythingV2, MODEL_IDS
from omegaconf import DictConfig, OmegaConf
import hydra
from utils.train_utils import *
from utils.trainer import train_one_epoch
from utils.validator import validate
import logging
from utils.logger import setup_logger



@hydra.main(config_path="config", config_name="base_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") and (not cfg.training.no_amp)
    wandb_run = None
    model_id = MODEL_IDS[cfg.model.model_id]
    print(f"Using model: {model_id}")
    print(f"dataset root: {cfg.dataset.dataset_root}")
    print(f"wandb project: {cfg.training.wandb_entity}/{cfg.training.wandb_project}")
    
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
        
    image_processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=None)
    dataset_kwargs = {
        "root_dir": cfg.dataset.dataset_root,
        "image_processor": image_processor,
        "image_size": (cfg.model.image_height, cfg.model.image_width),
        "min_depth": cfg.dataset.min_depth,
        "max_depth": cfg.dataset.max_depth,
        "min_valid_depth_ratio": cfg.dataset.min_valid_depth_ratio,
        "light_levels": LIGHT_LEVELS,
        "speed_levels": MOTION_LEVELS,
    }
    
    seen_val_topologies = [ str(topology).strip() for topology in cfg.dataset.seen_val_topologies ]
    unseen_val_topologies = [ str(topology).strip() for topology in cfg.dataset.unseen_val_topologies ]
    seen_val_topology_ids = {topology_id(topology) for topology in seen_val_topologies}
    unseen_val_topology_ids = {topology_id(topology) for topology in unseen_val_topologies}
    seen_topology_idx = torch.Tensor(list(seen_val_topology_ids)).long()
    unseen_topology_idx = torch.Tensor(list(unseen_val_topology_ids)).long()
    print("[Training] Seen validation topologies:", seen_topology_idx)
    print("[Training] Unseen validation topologies:", unseen_topology_idx)
    
    train_set = ATIRealWorldUncertaintyDataset(
        topologies=cfg.dataset.train_topologies,
        **dataset_kwargs,
    )
    val_set = ATIRealWorldUncertaintyValidationDataset(
        **dataset_kwargs,
    )
    copy_condition_normalization(val_set, train_set)
    validation_topology_counts = count_items_by_topology(val_set)
    seen_val_indices = topology_subset_indices(val_set, seen_val_topology_ids)
    unseen_val_indices = topology_subset_indices(val_set, unseen_val_topology_ids)
    seen_val_count = len(seen_val_indices)
    unseen_val_count = len(unseen_val_indices)
    
    dataset_metadata = {
        "condition_names": list(train_set.condition_names),
        "light_levels": list(train_set.light_levels),
        "speed_levels": list(train_set.speed_levels),
        "exposure_min": train_set.exposure_min,
        "exposure_max": train_set.exposure_max,
        "gain_min": train_set.gain_min,
        "gain_max": train_set.gain_max,
        "min_valid_depth_ratio": cfg.dataset.min_valid_depth_ratio,
        "train_scan_stats": dict(train_set.scan_stats),
        "validation_scan_stats": dict(val_set.scan_stats),
        "validation_topology_counts": dict(validation_topology_counts),
        "seen_validation_topologies": list(seen_val_topologies),
        "seen_validation_samples": seen_val_count,
        "unseen_validation_topologies": list(unseen_val_topologies),
        "unseen_validation_samples": unseen_val_count,
    }
    print(
        f"train samples: {len(train_set):,}, "
        f"val samples: {len(val_set):,}, "
        f"seen val samples: {seen_val_count:,}, "
        f"unseen val samples: {unseen_val_count:,}"
    )
    print(f"condition dim: {train_set.condition_dim}, names={list(train_set.condition_names)}")
    print(f"train scan stats: {train_set.scan_stats}")
    print(f"validation scan stats: {val_set.scan_stats}")
    print(f"validation topology counts: {validation_topology_counts}")
    
    if wandb_run is not None:
        wandb_run.config.update(dataset_metadata, allow_val_change=True)
    
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=pin_memory,
        collate_fn=ati_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=pin_memory,
        collate_fn=ati_collate_fn,
    )
    # seen_val_loader = DataLoader(
    #     Subset(val_set, seen_val_indices),
    #     batch_size=cfg.training.batch_size,
    #     shuffle=False,
    #     num_workers=cfg.dataset.num_workers,
    #     pin_memory=pin_memory,
    #     collate_fn=ati_collate_fn,
    # )
    # unseen_val_loader = DataLoader(
    #     Subset(val_set, unseen_val_indices),
    #     batch_size=cfg.training.batch_size,
    #     shuffle=False,
    #     num_workers=cfg.dataset.num_workers,
    #     pin_memory=pin_memory,
    #     collate_fn=ati_collate_fn,
    # )

    model = ConditionedGaussianDepthAnythingV2(
        model_id=model_id,
        cond_dim=train_set.condition_dim,
        freeze_backbone=cfg.training.freeze_backbone,
        min_log_var=cfg.training.min_log_var,
        max_log_var=cfg.training.max_log_var,
        uncertainty_width=cfg.model.uncertainty_width,
        uncertainty_blocks=cfg.model.uncertainty_blocks,
        uncertainty_dropout=cfg.model.uncertainty_dropout,
    ).to(device)
    
    backbone_params, uncertainty_params = count_model_parameters(model)
    param_groups = []
    if backbone_params:
        param_groups.append({"name": "backbone", "params": backbone_params, "lr": cfg.training.lr_backbone})
    if uncertainty_params:
        param_groups.append({"name": "uncertainty", "params": uncertainty_params, "lr": cfg.training.lr_uncertainty})
    if not param_groups:
        raise ValueError("No trainable parameters found")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.training.weight_decay)
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
            list_loss_weight=cfg.training.list_loss_weight,
            listnet_temperature=cfg.training.listnet_temperature,
            uncertainty_mode=cfg.training.uncertainty_mode,
            grad_clip=cfg.training.grad_clip,
            min_depth=cfg.dataset.min_depth,
            max_depth=cfg.dataset.max_depth,
            relative_align_mode=cfg.training.relative_align_mode,
            global_step=global_step,
            log_interval=cfg.training.log_interval,
        )
        val_total_metrics, val_seen_metrics, val_unseen_metrics = validate(
            epoch=epoch,
            model_id=model_id,
            model=model,
            loader=val_loader,
            device=device,
            amp=amp,
            seen_topology_numbers=seen_topology_idx,
            unseen_topology_numbers=unseen_topology_idx,
            lambda_smooth_logvar=cfg.training.lambda_smooth_logvar,
            list_loss_weight=cfg.training.list_loss_weight,
            listnet_temperature=cfg.training.listnet_temperature,
            uncertainty_mode=cfg.training.uncertainty_mode,
            correlation_max_samples=cfg.training.correlation_max_samples,
            min_depth=cfg.dataset.min_depth,
            max_depth=cfg.dataset.max_depth,
            relative_align_mode=cfg.training.relative_align_mode,
        )
        
        print(f"[epoch {epoch}] train={train_metrics}")
        print(f"[epoch {epoch}] val={val_total_metrics}")
        print(f"[epoch {epoch}] seen_val={val_seen_metrics}")
        print(f"[epoch {epoch}] unseen_val={val_unseen_metrics}")
        
        is_best = val_total_metrics["aggregated_abs_rel_unc_pearson"] > best_abs_rel_correlation
        if is_best:
            best_abs_rel_correlation = val_total_metrics["aggregated_abs_rel_unc_pearson"]
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
            wandb_run.summary["best_abs_rel_correlation"] = best_abs_rel_correlation
            wandb_run.summary["best_epoch"] = epoch

        epoch_log = {
            "epoch": epoch,
            "best/abs_rel_correlation": best_abs_rel_correlation,
            "best/is_best": int(is_best),
        }
        epoch_log.update({f"train/{key}": value for key, value in train_metrics.items()})
        epoch_log.update({f"val/{key}": value for key, value in val_total_metrics.items()})
        epoch_log.update({f"val_seen/{key}": value for key, value in val_seen_metrics.items()})
        epoch_log.update({f"val_unseen/{key}": value for key, value in val_unseen_metrics.items()})
        wandb_run.log(epoch_log, step=epoch)
    
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()


# seen_val_metrics = validate(
#             epoch=epoch,
#             model_id=model_id,
#             model=model,
#             loader=seen_val_loader,
#             device=device,
#             amp=amp,
#             lambda_smooth_logvar=cfg.training.lambda_smooth_logvar,
#             list_loss_weight=cfg.training.list_loss_weight,
#             listnet_temperature=cfg.training.listnet_temperature,
#             uncertainty_mode=cfg.training.uncertainty_mode,
#             correlation_max_samples=cfg.training.correlation_max_samples,
#             min_depth=cfg.dataset.min_depth,
#             max_depth=cfg.dataset.max_depth,
#             relative_align_mode=cfg.training.relative_align_mode,
#         )
#         unseen_val_metrics = validate(
#             epoch=epoch,
#             model_id=model_id,
#             model=model,
#             loader=unseen_val_loader,
#             device=device,
#             amp=amp,
#             lambda_smooth_logvar=cfg.training.lambda_smooth_logvar,
#             list_loss_weight=cfg.training.list_loss_weight,
#             listnet_temperature=cfg.training.listnet_temperature,
#             uncertainty_mode=cfg.training.uncertainty_mode,
#             correlation_max_samples=cfg.training.correlation_max_samples,
#             min_depth=cfg.dataset.min_depth,
#             max_depth=cfg.dataset.max_depth,
#             relative_align_mode=cfg.training.relative_align_mode,
#         )
