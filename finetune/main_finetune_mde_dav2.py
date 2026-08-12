from glob import glob
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from finetune.finetune_dataset import MDEFineTuningDataset, split_dataset
from finetune.utils import run_epoch, seed_everything


METRIC_MODEL_NAMES = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


@hydra.main(config_path="../config", config_name="base_finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    import wandb

    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = METRIC_MODEL_NAMES[cfg.model.model_id]

    csv_root = Path(cfg.dataset.csv_path)
    csv_paths = [csv_root] if csv_root.is_file() else sorted(
        map(Path, glob(str(csv_root / "*.csv")))
    )
    if not csv_paths:
        raise ValueError(f"No CSV files found in {csv_root}")

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    common = dict(
        csv_paths=csv_paths,
        image_processor=processor,
        camera_model_name="Orbbec",
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        path_replacements={
            "/media/michael/ssd1/AIoT_ATI/orbbec_realworld_dataset":
                cfg.dataset.dataset_root
        },
    )
    dataset = MDEFineTuningDataset(
        **common, topologies=cfg.dataset.topology_sets
    )
    train_set, val_set = split_dataset(
        dataset, cfg.dataset.training_ratio, cfg.training.seed
    )
    print(
        f"dataset={len(dataset):,} train={len(train_set):,} val={len(val_set):,} "
        f"topologies={list(cfg.dataset.topology_sets)}"
    )
    loader_args = dict(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataset.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr_finetune,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_scheduler_factor,
        patience=cfg.training.lr_scheduler_patience,
        threshold=cfg.training.lr_scheduler_threshold,
        cooldown=cfg.training.lr_scheduler_cooldown,
        min_lr=cfg.training.lr_finetune * cfg.training.lr_scheduler_min_lr_ratio,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    run = wandb.init(
        entity=cfg.training.wandb_entity,
        project=cfg.training.wandb_project,
        name=cfg.training.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    best_abs_rel = float("inf")
    output_dir = Path(cfg.dataset.output_dir)
    try:
        for epoch in range(1, cfg.training.num_epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device, optimizer, scaler,
                description=f"train {epoch:03d}/{cfg.training.num_epochs:03d}",
                min_depth=cfg.dataset.min_depth,
                max_depth=cfg.dataset.max_depth,
            )
            val_metrics = run_epoch(
                model, val_loader, device,
                description=f"val   {epoch:03d}/{cfg.training.num_epochs:03d}",
                min_depth=cfg.dataset.min_depth,
                max_depth=cfg.dataset.max_depth,
            )
            scheduler.step(val_metrics["abs_rel"])
            run.log(
                {
                    "mde_finetune_train/loss": train_metrics["loss"],
                    "mde_finetune_train/absl_rel": train_metrics["abs_rel"],
                    "mde_finetune_train/a1": train_metrics["a1"],
                    "mde_finetune_val/absl_rel": val_metrics["abs_rel"],
                    "mde_finetune_val/a1": val_metrics["a1"],
                },
                step=epoch,
                commit=True
            )
            print(
                f"epoch {epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_abs_rel={val_metrics['abs_rel']:.4f}"
            )
            if val_metrics["abs_rel"] < best_abs_rel:
                best_abs_rel = val_metrics["abs_rel"]
                model.save_pretrained(output_dir)
                processor.save_pretrained(output_dir)
    finally:
        run.finish()


if __name__ == "__main__":
    main()

