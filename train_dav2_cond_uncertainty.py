import argparse
import os

import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from transformers import AutoImageProcessor

from ati_dataset import (
    ATI_STATS_EXPOSURE_IDX,
    ATI_STATS_GAIN_IDX,
    ATI_STATS_LIGHT_LABEL_IDX,
    ATI_STATS_SPEED_LABEL_IDX,
    ATI_STATS_VALID_PIXEL_RATIO_IDX,
    ATIRealWorldDepthDataset,
    ati_collate_fn,
)
from correlation_utils import (
    compute_image_uncertainty_metric_correlations,
    compute_image_uncertainty_metric_values,
    compute_loss_uncertainty_correlations,
    compute_sparsification_ause_metrics,
)
from dav2_ati_model import ConditionedGaussianDepthAnythingV2, MODEL_IDS
from eval_utils import (
    compute_metrics,
    compute_relative_depth_metrics,
    _accumulate_finite_metrics,
    _mean_finite_metrics,
)
from loss_fn import gaussian_nll_depth_loss, image_level_listnet_loss


def _wandb_log_prefixed(wandb_run, prefix, metrics, step):
    if wandb_run is None:
        return

    wandb_run.log(
        {
            f"{prefix}/{key}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
        step=step,
    )


def _extend_image_metric_values(accumulator, image_values):
    for key, value in image_values.items():
        accumulator.setdefault(key, []).append(value.detach().float().cpu())


def _compute_global_image_correlations(accumulator):
    if not accumulator:
        empty = torch.empty(0)
        return compute_image_uncertainty_metric_correlations(
            {
                "mean_uncertainty": empty,
                "abs_rel": empty,
                "a1": empty,
            }
        )

    return compute_image_uncertainty_metric_correlations(
        {
            key: torch.cat(values, dim=0)
            for key, values in accumulator.items()
        }
    )

def _prefix_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}

def _optimizer_lr(optimizer, group_name):
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group["lr"]
    return 0.0


def _unpack_ati_batch(batch, device):
    (
        pixel_values,
        depth,
        valid_mask,
        condition,
        condition_stats,
    ) = batch

    return (
        pixel_values.to(device),
        depth.to(device),
        valid_mask.to(device),
        condition.to(device),
        condition_stats,
    )


def _condition_batch_metrics(condition_stats):
    return {
        "exposure_mean": float(condition_stats[:, ATI_STATS_EXPOSURE_IDX].mean().item()),
        "gain_mean": float(condition_stats[:, ATI_STATS_GAIN_IDX].mean().item()),
        "valid_pixel_ratio_mean": float(
            condition_stats[:, ATI_STATS_VALID_PIXEL_RATIO_IDX].mean().item()
        ),
        "light_label_mean": float(
            condition_stats[:, ATI_STATS_LIGHT_LABEL_IDX].mean().item()
        ),
        "speed_label_mean": float(
            condition_stats[:, ATI_STATS_SPEED_LABEL_IDX].mean().item()
        ),
    }


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
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    relative_align_mode: str = "scale_shift",
    wandb_run=None,
    global_step: int = 0,
    log_interval: int = 20,
    correlation_max_samples: int = 100_000,
):
    model.train()

    running_loss = 0.0
    running_nll_loss = 0.0
    running_list_loss = 0.0
    running_abs_rel = 0.0
    running_rmse = 0.0
    running_a1 = 0.0
    running_corr_samples = 0
    running_ause_samples = 0
    corr_sums = {}
    corr_counts = {}
    condition_sums = {}
    image_metric_values = {}
    processed_batches = 0

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        (
            pixel_values,
            depth,
            valid_mask,
            condition,
            condition_stats,
        ) = _unpack_ati_batch(batch, device)

        target_size = depth.shape[-2:]

        optimizer.zero_grad(set_to_none=True)

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

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if model_id.startswith("metric"):
            metrics = _prefix_metrics(
                "metric_depth",
                compute_metrics(out["mu"].detach(), depth, valid_mask)
            )
        else:
            metrics = _prefix_metrics(
                "relative_depth",
                compute_relative_depth_metrics(
                    out["mu"].detach(),
                    depth,
                    valid_mask,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    align_mode=relative_align_mode,
                ),
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
        batch_image_values = compute_image_uncertainty_metric_values(
            out["mu"].detach(),
            depth,
            valid_mask,
            uncertainty=out["std"].detach(),
        )
        batch_image_correlations = compute_image_uncertainty_metric_correlations(batch_image_values)
        batch_condition_metrics = _condition_batch_metrics(condition_stats)

        uncertainty_metrics = {
            **correlations,
            **ause_metrics,
            **batch_image_correlations,
        }
        epoch_mean_metrics = {
            **correlations,
            **ause_metrics,
        }

        prefix_head = "metric_depth" if model_id.startswith("metric") else "relative_depth"
        running_loss += loss.item()
        running_nll_loss += nll_loss.item()
        running_list_loss += list_loss.item()
        running_abs_rel += metrics[f"{prefix_head}_abs_rel"]
        running_rmse += metrics[f"{prefix_head}_rmse"]
        running_a1 += metrics[f"{prefix_head}_a1"]
        running_corr_samples += correlations["loss_uncertainty_samples"]
        running_ause_samples += ause_metrics["ause_samples"]
        _accumulate_finite_metrics(corr_sums, corr_counts, epoch_mean_metrics)
        _extend_image_metric_values(image_metric_values, batch_image_values)
        processed_batches += 1

        for key, value in batch_condition_metrics.items():
            condition_sums[key] = condition_sums.get(key, 0.0) + value

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"[train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"abs_rel={metrics[f'{prefix_head}_abs_rel']:.4f} "
                f"a1={metrics[f'{prefix_head}_a1']:.4f} "
                f"exposure={batch_condition_metrics['exposure_mean']:.1f} "
                f"gain={batch_condition_metrics['gain_mean']:.1f} "
                f"valid_ratio={batch_condition_metrics['valid_pixel_ratio_mean']:.3f} "
                f"ause_abs_rel={ause_metrics['ause_abs_rel']:.4f}"
            )
            train_step_metrics = {
                "loss_step": loss.item(),
                "nll_loss_step": nll_loss.item(),
                "list_loss_step": list_loss.item(),
                "abs_rel_step": metrics[f"{prefix_head}_abs_rel"],
                "rmse_step": metrics[f"{prefix_head}_rmse"],
                "a1_step": metrics[f"{prefix_head}_a1"],
                "epoch": epoch,
                "lr_backbone": _optimizer_lr(optimizer, "backbone"),
                "lr_uncertainty": _optimizer_lr(optimizer, "uncertainty"),
                **{f"{key}_step": value for key, value in batch_condition_metrics.items()},
                **{f"{key}_step": value for key, value in uncertainty_metrics.items()},
            }
            _wandb_log_prefixed(wandb_run, "train", train_step_metrics, global_step)

        global_step += 1

    n = max(processed_batches, 1)
    epoch_metrics = {
        "loss": running_loss / n,
        "nll_loss": running_nll_loss / n,
        "list_loss": running_list_loss / n,
        "abs_rel": running_abs_rel / n,
        "rmse": running_rmse / n,
        "a1": running_a1 / n,
        "loss_uncertainty_samples": running_corr_samples,
        "ause_samples": running_ause_samples,
    }
    epoch_metrics.update({key: value / n for key, value in condition_sums.items()})
    epoch_metrics.update(_mean_finite_metrics(corr_sums, corr_counts))
    epoch_metrics.update(_compute_global_image_correlations(image_metric_values))

    return epoch_metrics, global_step


@torch.no_grad()
def validate(
    model_id: str,
    model,
    loader,
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

    running_loss = 0.0
    running_abs_rel = 0.0
    running_rmse = 0.0
    running_a1 = 0.0
    running_corr_samples = 0
    running_ause_samples = 0
    corr_sums = {}
    corr_counts = {}
    condition_sums = {}
    image_metric_values = {}
    processed_batches = 0

    for batch in loader:
        if batch is None:
            continue

        (
            pixel_values,
            depth,
            valid_mask,
            condition,
            condition_stats,
        ) = _unpack_ati_batch(batch, device)

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

        if model_id.startswith("metric"):
            metrics = _prefix_metrics(
                "metric_depth",
                compute_metrics(out["mu"].detach(), depth, valid_mask)
            )
        else:
            metrics = _prefix_metrics(
                "relative_depth",
                compute_relative_depth_metrics(
                    out["mu"].detach(),
                    depth,
                    valid_mask,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    align_mode=relative_align_mode,
                ),
            )
            
        correlations = compute_loss_uncertainty_correlations(
            out["mu"],
            out["log_var"],
            depth,
            valid_mask,
            uncertainty=out["std"],
            max_samples=correlation_max_samples,
        )
        ause_metrics = compute_sparsification_ause_metrics(
            out["mu"],
            depth,
            valid_mask,
            uncertainty=out["std"],
            max_samples=correlation_max_samples,
        )
        batch_image_values = compute_image_uncertainty_metric_values(
            out["mu"],
            depth,
            valid_mask,
            uncertainty=out["std"],
        )
        batch_condition_metrics = _condition_batch_metrics(condition_stats)
        epoch_mean_metrics = {
            **correlations,
            **ause_metrics,
        }

        prefix_head = "metric_depth" if model_id.startswith("metric") else "relative_depth"
        running_loss += loss.item()
        running_abs_rel += metrics[f"{prefix_head}_abs_rel"]
        running_rmse += metrics[f"{prefix_head}_rmse"]
        running_a1 += metrics[f"{prefix_head}_a1"]
        running_corr_samples += correlations["loss_uncertainty_samples"]
        running_ause_samples += ause_metrics["ause_samples"]
        _accumulate_finite_metrics(corr_sums, corr_counts, epoch_mean_metrics)
        _extend_image_metric_values(image_metric_values, batch_image_values)
        processed_batches += 1

        for key, value in batch_condition_metrics.items():
            condition_sums[key] = condition_sums.get(key, 0.0) + value

    n = max(processed_batches, 1)
    val_metrics = {
        "loss": running_loss / n,
        "abs_rel": running_abs_rel / n,
        "rmse": running_rmse / n,
        "a1": running_a1 / n,
        "loss_uncertainty_samples": running_corr_samples,
        "ause_samples": running_ause_samples,
    }
    val_metrics.update({key: value / n for key, value in condition_sums.items()})
    val_metrics.update(_mean_finite_metrics(corr_sums, corr_counts))
    val_metrics.update(_compute_global_image_correlations(image_metric_values))

    return val_metrics


def save_checkpoint(model, image_processor, output_dir, epoch, val_metrics, dataset_metadata):
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, "pytorch_model_ati_cond_uncertainty.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "dataset_metadata": dataset_metadata,
        },
        ckpt_path,
    )

    image_processor.save_pretrained(output_dir)

    print(f"saved checkpoint to: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, default="/media/michael/ssd1/AIoT_ATI/realworld_dataset")
    parser.add_argument("--model", type=str, default="metric-indoor-small", choices=list(MODEL_IDS.keys()))
    parser.add_argument("--output_dir", type=str, default="./da2_ati_cond_uncertainty_ckpt")

    parser.add_argument("--image_height", type=int, default=518)
    parser.add_argument("--image_width", type=int, default=518)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_backbone", type=float, default=1e-6)
    parser.add_argument("--lr_uncertainty", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_valid_depth_ratio", type=float, default=0.3)
    parser.add_argument("--min_log_var", type=float, default=-5.0)
    parser.add_argument("--max_log_var", type=float, default=3.0)
    parser.add_argument("--uncertainty_width", type=int, default=64)
    parser.add_argument("--uncertainty_blocks", type=int, default=6)
    parser.add_argument("--uncertainty_dropout", type=float, default=0.05)

    parser.add_argument("--lambda_smooth_logvar", type=float, default=1e-3)
    parser.add_argument("--list_loss_weight", type=float, default=0.1)
    parser.add_argument("--listnet_temperature", type=float, default=0.1)
    parser.add_argument("--uncertainty_mode", type=str, default="top20", choices=["mean", "top10", "top20"])
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--light_levels", nargs="*", default=["dark", "dim", "normal"])
    parser.add_argument("--speed_levels", nargs="*", default=["slow", "fast"])
    parser.add_argument("--scene_prefixes", nargs="*", default=["comlab_scene2", "realsense_scene"])

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--correlation_max_samples", type=int, default=100_000)
    parser.add_argument("--relative_align_mode", type=str, default="scale_shift", choices=["median", "scale_shift"])
    parser.add_argument("--wandb_entity", type=str, default="artificial_tripartite_intelligence_team")
    parser.add_argument("--wandb_project", type=str, default="mde_uncertainty_measure")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") and (not args.no_amp)
    wandb_run = None

    model_id = MODEL_IDS[args.model]
    print(f"Using model: {model_id}")
    print(f"dataset root: {args.dataset_root}")
    print(f"wandb project: {args.wandb_entity}/{args.wandb_project}")

    if not args.disable_wandb:
        if wandb is None:
            raise ImportError(
                "wandb is required for logging. Install it with `pip install wandb` "
                "or pass `--disable_wandb` for a local run without wandb."
            )
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    image_processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=args.hf_cache_dir)

    dataset_kwargs = {
        "root_dir": args.dataset_root,
        "image_processor": image_processor,
        "image_size": (args.image_height, args.image_width),
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "min_valid_depth_ratio": args.min_valid_depth_ratio,
        "light_levels": args.light_levels,
        "speed_levels": args.speed_levels,
        "scene_prefixes": args.scene_prefixes,
    }
    train_set = ATIRealWorldDepthDataset(
        split="train",
        max_samples=args.max_train_samples,
        **dataset_kwargs,
    )
    val_set = ATIRealWorldDepthDataset(
        split="validation",
        max_samples=args.max_val_samples,
        **dataset_kwargs,
    )

    dataset_metadata = {
        "condition_names": list(train_set.condition_names),
        "light_levels": list(train_set.light_levels),
        "speed_levels": list(train_set.speed_levels),
        "exposure_min": train_set.exposure_min,
        "exposure_max": train_set.exposure_max,
        "gain_min": train_set.gain_min,
        "gain_max": train_set.gain_max,
        "min_valid_depth_ratio": args.min_valid_depth_ratio,
    }
    print(f"train samples: {len(train_set):,}, val samples: {len(val_set):,}")
    print(f"condition dim: {train_set.condition_dim}, names={list(train_set.condition_names)}")

    if wandb_run is not None:
        wandb_run.config.update(dataset_metadata, allow_val_change=True)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=ati_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=ati_collate_fn,
    )

    model = ConditionedGaussianDepthAnythingV2(
        model_id=model_id,
        cond_dim=train_set.condition_dim,
        cache_dir=args.hf_cache_dir,
        freeze_backbone=args.freeze_backbone,
        min_log_var=args.min_log_var,
        max_log_var=args.max_log_var,
        uncertainty_width=args.uncertainty_width,
        uncertainty_blocks=args.uncertainty_blocks,
        uncertainty_dropout=args.uncertainty_dropout,
    ).to(device)

    backbone_params = []
    uncertainty_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("uncertainty_head"):
            uncertainty_params.append(param)
        else:
            backbone_params.append(param)

    print(
        "trainable params: "
        f"backbone={sum(p.numel() for p in backbone_params):,}, "
        f"uncertainty={sum(p.numel() for p in uncertainty_params):,}"
    )

    param_groups = []
    if backbone_params:
        param_groups.append({"name": "backbone", "params": backbone_params, "lr": args.lr_backbone})
    if uncertainty_params:
        param_groups.append({"name": "uncertainty", "params": uncertainty_params, "lr": args.lr_uncertainty})
    if not param_groups:
        raise ValueError("No trainable parameters found")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_abs_rel = float("inf")
    best_abs_rel_correlation = float("-inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model_id=model_id,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp=amp,
            lambda_smooth_logvar=args.lambda_smooth_logvar,
            list_loss_weight=args.list_loss_weight,
            listnet_temperature=args.listnet_temperature,
            uncertainty_mode=args.uncertainty_mode,
            grad_clip=args.grad_clip,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            relative_align_mode=args.relative_align_mode,
            wandb_run=wandb_run,
            global_step=global_step,
            log_interval=args.log_interval,
            correlation_max_samples=args.correlation_max_samples,
        )

        val_metrics = validate(
            model_id=model_id,
            model=model,
            loader=val_loader,
            device=device,
            amp=amp,
            lambda_smooth_logvar=args.lambda_smooth_logvar,
            list_loss_weight=args.list_loss_weight,
            listnet_temperature=args.listnet_temperature,
            uncertainty_mode=args.uncertainty_mode,
            correlation_max_samples=args.correlation_max_samples,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            relative_align_mode=args.relative_align_mode,
        )

        print(f"[epoch {epoch}] train={train_metrics}")
        print(f"[epoch {epoch}] val={val_metrics}")

        # is_best = val_metrics["abs_rel"] < best_abs_rel
        is_best = val_metrics["image_mean_uncertainty_abs_rel_pearson"] > best_abs_rel_correlation
        if is_best:
            best_abs_rel_correlation = val_metrics["image_mean_uncertainty_abs_rel_pearson"]
            save_checkpoint(
                model,
                image_processor,
                args.output_dir,
                epoch,
                val_metrics,
                dataset_metadata,
            )
            if wandb_run is not None:
                wandb_run.summary["best_abs_rel_correlation"] = best_abs_rel_correlation
                wandb_run.summary["best_epoch"] = epoch

        if wandb_run is not None:
            epoch_log = {
                "epoch": epoch,
                "best/abs_rel_correlation": best_abs_rel_correlation,
                "best/is_best": int(is_best),
            }
            epoch_log.update({f"train/{key}": value for key, value in train_metrics.items()})
            epoch_log.update({f"val/{key}": value for key, value in val_metrics.items()})
            wandb_run.log(epoch_log, step=global_step)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
