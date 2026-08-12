from functools import partial
from glob import glob
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from finetune.finetune_dataset import MDEFineTuningDataset, split_dataset
from finetune.utils import (
    metric3d_decoder_only,
    predict_metric3d_decoder_only,
    run_epoch,
    seed_everything,
)


@hydra.main(config_path="../config", config_name="base_finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    import wandb

    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_root = Path(cfg.dataset.csv_path)
    csv_paths = [csv_root] if csv_root.is_file() else sorted(
        map(Path, glob(str(csv_root / "*.csv")))
    )
    if not csv_paths:
        raise ValueError(f"No CSV files found in {csv_root}")

    dataset = MDEFineTuningDataset(
        csv_paths=csv_paths,
        image_processor=None,
        image_size=(cfg.model.image_height, cfg.model.image_width),
        camera_model_name="Orbbec",
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        topologies=cfg.dataset.topology_sets,
        path_replacements={
            "/media/michael/ssd1/AIoT_ATI/orbbec_realworld_dataset":
                cfg.dataset.dataset_root
        },
    )
    train_set, val_set = split_dataset(
        dataset, cfg.dataset.training_ratio, cfg.training.seed
    )
    print(
        f"dataset={len(dataset):,} train={len(train_set):,} val={len(val_set):,} "
        f"topologies={list(cfg.dataset.topology_sets)}"
    )
    loader_args = dict(
        batch_size=cfg.training.metric3d_batch_size,
        num_workers=cfg.dataset.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = torch.hub.load(
        "yvanyin/metric3d",
        cfg.evaluation.metric3d_model,
        pretrain=True,
        trust_repo=True,
    ).to(device)
    decoder = metric3d_decoder_only(model)
    trainable = sum(parameter.numel() for parameter in decoder.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"decoder-only trainable parameters: {trainable:,}/{total:,}")
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
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
    metric3d_predictor = partial(
        predict_metric3d_decoder_only,
        focal_length=cfg.evaluation.metric3d_focal_length,
    )
    run = wandb.init(
        entity=cfg.training.wandb_entity,
        project=cfg.training.wandb_project,
        name=f"{cfg.training.wandb_name}_metric3dv2",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    best_abs_rel = float("inf")
    output_dir = Path(cfg.model.metric3d_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(1, cfg.training.num_epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device, optimizer, scaler,
                predict_depth=metric3d_predictor,
                description=f"train {epoch:03d}/{cfg.training.num_epochs:03d}",
                min_depth=cfg.dataset.min_depth,
                max_depth=cfg.dataset.max_depth,
            )
            val_metrics = run_epoch(
                model, val_loader, device,
                predict_depth=metric3d_predictor,
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
                torch.save(model.state_dict(), output_dir / "model.pth")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
