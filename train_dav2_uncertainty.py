import os
import argparse
import math
import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from transformers import AutoImageProcessor
from dav2_model import GaussianDepthAnythingV2, MODEL_IDS
from dataset import NYUv2RGBDepthDataset, nyu_collate_fn
from correlation_utils import (
    compute_image_uncertainty_metric_correlations,
    compute_image_uncertainty_metric_values,
    compute_loss_uncertainty_correlations,
    compute_sparsification_ause_metrics,
)
from loss_fn import gaussian_nll_depth_loss, image_level_listnet_loss
from eval_utils import (
    compute_metrics,
    compute_relative_depth_metrics,
    _accumulate_finite_metrics,
    _mean_finite_metrics,
)

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


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    amp: bool,
    lambda_smooth_logvar: float,
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
    running_abs_rel = 0.0
    running_rmse = 0.0
    running_a1 = 0.0
    running_corr_samples = 0
    running_ause_samples = 0
    corr_sums = {}
    corr_counts = {}
    image_metric_values = {}

    for step, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        depth = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        target_size = depth.shape[-2:]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=amp):
            out = model(pixel_values, target_size=target_size)
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
                temperature=0.1,
                uncertainty_mode="top20",
            )
            loss = nll_loss + 0.1 * list_loss

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        metrics = compute_metrics(out["mu"].detach(), depth, valid_mask)
        relative_metrics = _prefix_metrics(
            "relative",
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
        uncertainty_metrics = {
            **correlations,
            **ause_metrics,
            **batch_image_correlations,
        }
        epoch_mean_metrics = {
            **relative_metrics,
            **correlations,
            **ause_metrics,
        }

        running_loss += loss.item()
        running_abs_rel += metrics["abs_rel"]
        running_rmse += metrics["rmse"]
        running_a1 += metrics["a1"]
        running_corr_samples += correlations["loss_uncertainty_samples"]
        running_ause_samples += ause_metrics["ause_samples"]
        _accumulate_finite_metrics(corr_sums, corr_counts, epoch_mean_metrics)
        _extend_image_metric_values(image_metric_values, batch_image_values)

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"[train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"abs_rel={metrics['abs_rel']:.4f} "
                f"a1={metrics['a1']:.4f} "
                f"relative_abs_rel={relative_metrics['relative_abs_rel']:.4f} "
                f"loss_uncertainty_pearson={correlations['loss_uncertainty_pearson']:.4f} "
                f"ause_abs_rel={ause_metrics['ause_abs_rel']:.4f}"
            )
            train_step_metrics = {
                "loss_step": loss.item(),
                "abs_rel_step": metrics["abs_rel"],
                "rmse_step": metrics["rmse"],
                "a1_step": metrics["a1"],
                "epoch": epoch,
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "lr_uncertainty": optimizer.param_groups[1]["lr"],
                **{f"{key}_step": value for key, value in relative_metrics.items()},
                **{f"{key}_step": value for key, value in uncertainty_metrics.items()},
            }
            _wandb_log_prefixed(wandb_run, "train", train_step_metrics, global_step)

        global_step += 1

    n = max(len(loader), 1)
    epoch_metrics = {
        "loss": running_loss / n,
        "abs_rel": running_abs_rel / n,
        "rmse": running_rmse / n,
        "a1": running_a1 / n,
        "loss_uncertainty_samples": running_corr_samples,
        "ause_samples": running_ause_samples,
    }
    epoch_metrics.update(_mean_finite_metrics(corr_sums, corr_counts))
    epoch_metrics.update(_compute_global_image_correlations(image_metric_values))

    return epoch_metrics, global_step


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    amp: bool,
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
    image_metric_values = {}

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        depth = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        target_size = depth.shape[-2:]

        with torch.autocast(device_type="cuda", enabled=amp):
            out = model(pixel_values, target_size=target_size)
            loss = gaussian_nll_depth_loss(
                out["mu"],
                out["log_var"],
                depth,
                valid_mask,
            )

        metrics = compute_metrics(out["mu"], depth, valid_mask)
        relative_metrics = _prefix_metrics(
            "relative",
            compute_relative_depth_metrics(
                out["mu"],
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
        epoch_mean_metrics = {
            **relative_metrics,
            **correlations,
            **ause_metrics,
        }

        running_loss += loss.item()
        running_abs_rel += metrics["abs_rel"]
        running_rmse += metrics["rmse"]
        running_a1 += metrics["a1"]
        running_corr_samples += correlations["loss_uncertainty_samples"]
        running_ause_samples += ause_metrics["ause_samples"]
        _accumulate_finite_metrics(corr_sums, corr_counts, epoch_mean_metrics)
        _extend_image_metric_values(image_metric_values, batch_image_values)

    n = max(len(loader), 1)
    val_metrics = {
        "loss": running_loss / n,
        "abs_rel": running_abs_rel / n,
        "rmse": running_rmse / n,
        "a1": running_a1 / n,
        "loss_uncertainty_samples": running_corr_samples,
        "ause_samples": running_ause_samples,
    }
    val_metrics.update(_mean_finite_metrics(corr_sums, corr_counts))
    val_metrics.update(_compute_global_image_correlations(image_metric_values))

    return val_metrics


def save_checkpoint(model, image_processor, output_dir, epoch, val_metrics):
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, "pytorch_model_uncertainty.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
        },
        ckpt_path,
    )

    # HF processor도 같이 저장
    image_processor.save_pretrained(output_dir)

    print(f"saved checkpoint to: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="metric-indoor-small", choices=list(MODEL_IDS.keys()))
    # parser.add_argument("--train_csv", type=str, required=True)
    # parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./da2_gaussian_nll_ckpt")

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

    parser.add_argument("--lambda_smooth_logvar", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)

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

    image_processor = AutoImageProcessor.from_pretrained(model_id)

    image_size = (args.image_height, args.image_width)

    train_set = NYUv2RGBDepthDataset(
        split="train",
        image_processor=image_processor,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        cache_dir=args.hf_cache_dir,
    )

    val_set = NYUv2RGBDepthDataset(
        split="validation",
        image_processor=image_processor,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        cache_dir=args.hf_cache_dir,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=nyu_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=nyu_collate_fn,
    )

    model = GaussianDepthAnythingV2(
        model_id=model_id,
        freeze_backbone=args.freeze_backbone,
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

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": uncertainty_params, "lr": args.lr_uncertainty},
        ],
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_abs_rel = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp=amp,
            lambda_smooth_logvar=args.lambda_smooth_logvar,
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
            model=model,
            loader=val_loader,
            device=device,
            amp=amp,
            correlation_max_samples=args.correlation_max_samples,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            relative_align_mode=args.relative_align_mode,
        )

        print(f"[epoch {epoch}] train={train_metrics}")
        print(f"[epoch {epoch}] val={val_metrics}")

        is_best = val_metrics["abs_rel"] < best_abs_rel
        if is_best:
            best_abs_rel = val_metrics["abs_rel"]
            save_checkpoint(model, image_processor, args.output_dir, epoch, val_metrics)
            if wandb_run is not None:
                wandb_run.summary["best_abs_rel"] = best_abs_rel
                wandb_run.summary["best_epoch"] = epoch

        if wandb_run is not None:
            epoch_log = {
                "epoch": epoch,
                "best/abs_rel": best_abs_rel,
                "best/is_best": int(is_best),
            }
            epoch_log.update({f"train/{key}": value for key, value in train_metrics.items()})
            epoch_log.update({f"val/{key}": value for key, value in val_metrics.items()})
            wandb_run.log(epoch_log, step=global_step)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
