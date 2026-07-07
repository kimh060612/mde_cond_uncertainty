from scipy.fftpack import shift
from typing import Dict

from dataset.ati_dataset_refactored import (
    ATIRealWorldUncertaintyBaseDataset
)
import torch
import os
from evaluation_utils.eval_utils import compute_relative_alignment, ensure_bchw
import random
import numpy as np


def seed_everything(
    seed: int = 42,
    deterministic: bool = True,
) -> None:
    """
    Fix random seeds for reproducible Python, NumPy, and PyTorch experiments.

    Args:
        seed:
            Random seed value.

        deterministic:
            If True, enables deterministic PyTorch algorithms where possible.
            This can reduce performance and may raise errors for operations
            without deterministic implementations.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuBLAS reproducibility. Must be set before relevant CUDA operations.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

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
    return compute_relative_alignment(
        pred,
        gt,
        valid_mask,
        align_mode="scale_shift",
        eps=eps,
    )

def reshape_group_batch(tensor: torch.Tensor, num_groups: int, num_candidates: int) -> torch.Tensor:
    return tensor.reshape(
        num_groups,
        num_candidates,
        *tensor.shape[1:],
    )
    
def masked_image_mean(values, valid_mask):
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
    
def tensor_device(
    tensor_dict: Dict[str, torch.Tensor], 
    device: torch.device
) -> Dict[str, torch.Tensor]:
    
    t_dict = { key: tensor.to(device) for key, tensor in tensor_dict.items() if isinstance(tensor, torch.Tensor) }
    non_tensor_dict = { key: tensor for key, tensor in tensor_dict.items() if not isinstance(tensor, torch.Tensor) }
    return {
        **t_dict, 
        **non_tensor_dict
    }