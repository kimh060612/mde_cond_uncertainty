from scipy.fftpack import shift

from dataset.ati_dataset_refactored import (
    ATIRealWorldUncertaintyBaseDataset
)
import torch
import os
from evaluation_utils.eval_utils import ensure_bchw

def prefix_metrics(prefix, metrics):
    return {f"{prefix}/{key}": value for key, value in metrics.items()}

def unpack_ati_batch(batch, device):
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

def copy_condition_normalization(
    target_dataset: ATIRealWorldUncertaintyBaseDataset, 
    source_dataset: ATIRealWorldUncertaintyBaseDataset
):
    target_dataset.exposure_min = source_dataset.exposure_min
    target_dataset.exposure_max = source_dataset.exposure_max
    target_dataset.gain_min = source_dataset.gain_min
    target_dataset.gain_max = source_dataset.gain_max
    
def topology_id(topology: str) -> int:
    topology = str(topology).strip()
    topology_suffix = topology[len("topology"):]
    if not topology_suffix.isdigit(): raise ValueError(f"Expected numeric topology name, got {topology}")
    return int(topology_suffix)
    
def count_items_by_topology(dataset):
    counts = {}
    for item in dataset.items:
        counts[item.topology] = counts.get(item.topology, 0) + 1
    return counts

def topology_subset_indices(
    dataset: ATIRealWorldUncertaintyBaseDataset, 
    topology_ids
):
    return [
        idx
        for idx, item in enumerate(dataset.items)
        if topology_id(item.topology) in topology_ids
    ]

def count_model_parameters(model: torch.nn.Module):
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
    return backbone_params, uncertainty_params

def count_model_param_finetune(model: torch.nn.Module):
    num_trainable_backbone = sum(
        parameter.numel()
        for parameter in model.depth_model.backbone.parameters()
        if parameter.requires_grad
    )

    num_trainable_decoder = sum(
        parameter.numel()
        for module in [
            model.depth_model.neck,
            model.depth_model.head,
        ]
        for parameter in module.parameters()
        if parameter.requires_grad
    )

    print("Trainable backbone:", num_trainable_backbone)
    print("Trainable decoder:", num_trainable_decoder)

def save_checkpoint(model, image_processor, output_dir, epoch, val_metrics, dataset_metadata):
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, f"ckpt_model_epoch{epoch}.pt")
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


@torch.no_grad()
def compute_align_scale_shift(pred, gt, valid_mask, eps=1e-8):
    pred = ensure_bchw(pred) # [B, 1, H, W]
    gt = ensure_bchw(gt) # [B, 1, H, W]
    valid_mask = ensure_bchw(valid_mask).bool()
    calc_dtype = torch.float64 if pred.dtype == torch.float64 or gt.dtype == torch.float64 else torch.float32
    pred = pred.to(dtype=calc_dtype)
    gt = gt.to(dtype=calc_dtype)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: target {gt.shape}, pred {pred.shape}")
    if valid_mask.shape != pred.shape:
        valid_mask = valid_mask.expand_as(pred)

    eps = 1e-8    
    relative_mask = valid_mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0) & (pred > 0)
    gt_inv = 1.0 / (gt + eps)
    x = torch.where(relative_mask, pred, torch.zeros_like(pred))
    y = torch.where(relative_mask, gt_inv, torch.zeros_like(gt_inv))

    valid_counts = relative_mask.flatten(1).sum(dim=1).to(dtype=calc_dtype)
    safe_counts = valid_counts.clamp_min(1.0)
    sum_x = x.flatten(1).sum(dim=1)
    sum_y = y.flatten(1).sum(dim=1)
    sum_xx = (x * x).flatten(1).sum(dim=1)
    sum_xy = (x * y).flatten(1).sum(dim=1)

    denom = safe_counts * sum_xx - sum_x.square()
    stable = (valid_counts > 1) & (denom.abs() > eps)
    scale = torch.where(
        stable,
        (safe_counts * sum_xy - sum_x * sum_y) / denom.clamp(min=eps),
        torch.zeros_like(valid_counts),
    )
    shift = torch.where(
        valid_counts > 0,
        (sum_y - scale * sum_x) / safe_counts,
        torch.zeros_like(valid_counts),
    )
    scale_map = scale[:, None, None, None]
    shift_map = shift[:, None, None, None]
    return scale_map, shift_map
