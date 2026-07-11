from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.ati_dataset_refactored import (  # noqa: E402
    ATIRealWorldUncertaintyValidationDataset,
    LIGHT_LEVELS,
    MOTION_LEVELS,
    ati_collate_fn,
)
from evaluation_utils.eval_metrics import (  # noqa: E402
    compute_comprehensive_depth_metrics,
    compute_vector_masked_correlations,
)
from evaluation_utils.eval_utils import (  # noqa: E402
    align_relative_prediction_to_depth_space,
    depth_error_maps,
    ensure_bchw,
)


MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    "metric-indoor-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-indoor-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "metric-indoor-large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "metric-outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "metric-outdoor-base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "metric-outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


@dataclass
class CorrelationAccumulator:
    image_uncertainty: list[torch.Tensor] = field(default_factory=list)
    image_abs_rel: list[torch.Tensor] = field(default_factory=list)
    pixel_uncertainty: list[torch.Tensor] = field(default_factory=list)
    pixel_abs_rel: list[torch.Tensor] = field(default_factory=list)

    def extend_images(
        self,
        uncertainty: torch.Tensor,
        abs_rel: torch.Tensor,
    ) -> None:
        self.image_uncertainty.append(uncertainty.detach().cpu().float().flatten())
        self.image_abs_rel.append(abs_rel.detach().cpu().float().flatten())

    def extend_pixels(
        self,
        uncertainty: torch.Tensor,
        abs_rel_error: torch.Tensor,
        valid_mask: torch.Tensor,
        max_pixels: int | None,
    ) -> None:
        uncertainty = uncertainty.detach().flatten().float()
        abs_rel_error = abs_rel_error.detach().flatten().float()
        valid_mask = valid_mask.detach().flatten().bool()
        valid_mask = (
            valid_mask
            & torch.isfinite(uncertainty)
            & torch.isfinite(abs_rel_error)
        )
        if not valid_mask.any():
            return

        uncertainty = uncertainty[valid_mask].cpu()
        abs_rel_error = abs_rel_error[valid_mask].cpu()
        if max_pixels is not None and max_pixels > 0 and uncertainty.numel() > max_pixels:
            indices = uniform_indices(uncertainty.numel(), max_pixels, uncertainty.device)
            uncertainty = uncertainty[indices]
            abs_rel_error = abs_rel_error[indices]

        self.pixel_uncertainty.append(uncertainty)
        self.pixel_abs_rel.append(abs_rel_error)

    def tensor(self, name: str) -> torch.Tensor:
        values = getattr(self, name)
        if not values:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(values, dim=0).float()


class GradientUncertaintyExtractor:
    """
    Backward-hook based implementation of gradient uncertainty from GrUMoDepth.

    Hooks collect dL_aux / d(feature_map). Each layer map is channel-reduced,
    resized to the depth map resolution, min-max normalised per image, then
    merged across selected layers.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Sequence[str],
        channel_reduction: str,
        layer_reduction: str,
        eps: float = 1e-12,
    ) -> None:
        self.model = model
        self.layer_names = tuple(layer_names)
        self.channel_reduction = channel_reduction
        self.layer_reduction = layer_reduction
        self.eps = eps
        self.gradients: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

        modules = dict(model.named_modules())
        missing = [name for name in self.layer_names if name not in modules]
        if missing:
            available = ", ".join(decoder_layer_candidates(model)[-20:])
            raise ValueError(
                f"Unknown layer(s): {missing}. Recent decoder candidates: {available}"
            )

        for name in self.layer_names:
            self.handles.append(
                modules[name].register_full_backward_hook(self._hook(name))
            )

    def _hook(self, name: str):
        def capture_gradient(
            module: nn.Module,
            grad_input: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor | None, ...],
        ) -> None:
            del module, grad_input
            if not grad_output or grad_output[0] is None:
                return
            self.gradients[name] = grad_output[0].detach()

        return capture_gradient

    def clear(self) -> None:
        self.gradients.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def uncertainty_map(self, target_size: tuple[int, int]) -> torch.Tensor:
        missing = [name for name in self.layer_names if name not in self.gradients]
        if missing:
            raise RuntimeError(
                "No gradients were captured for "
                f"{missing}. Check that the selected layers affect the depth output."
            )

        maps = [
            self._layer_uncertainty(self.gradients[name], target_size)
            for name in self.layer_names
        ]
        stacked = torch.stack(maps, dim=0)
        if len(maps) == 1:
            return stacked.squeeze(0)
        if self.layer_reduction == "max":
            return stacked.max(dim=0).values
        if self.layer_reduction == "mean":
            return stacked.mean(dim=0)
        if self.layer_reduction == "std":
            return stacked.std(dim=0, unbiased=False)
        raise ValueError(f"Unknown layer_reduction: {self.layer_reduction}")

    def _layer_uncertainty(
        self,
        gradient: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if gradient.ndim != 4:
            raise ValueError(
                f"Expected BCHW gradient tensor, got {tuple(gradient.shape)}"
            )

        gradient = gradient.float()
        if self.channel_reduction == "max_abs":
            reduced = gradient.abs().amax(dim=1, keepdim=True)
        elif self.channel_reduction == "norm":
            reduced = torch.linalg.vector_norm(gradient, ord=2, dim=1, keepdim=True)
        elif self.channel_reduction == "mean_abs":
            reduced = gradient.abs().mean(dim=1, keepdim=True)
        elif self.channel_reduction == "sum_abs":
            reduced = gradient.abs().sum(dim=1, keepdim=True)
        elif self.channel_reduction == "max":
            reduced = gradient.max(dim=1, keepdim=True).values
        else:
            raise ValueError(f"Unknown channel_reduction: {self.channel_reduction}")

        if reduced.shape[-2:] != target_size:
            reduced = F.interpolate(
                reduced,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return minmax_normalize_per_image(reduced, eps=self.eps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GrUMoDepth-style gradient-based uncertainty on the ATI "
            "validation dataset and save AbsRel correlations to CSV."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing val_comlab_scene_* directories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("evaluation_models") / "gradient_base_unc_correlation.csv",
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="small",
        help="Depth Anything V2 alias from this script or a full Hugging Face model id.",
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        default=None,
        help="Optional image processor directory. Defaults to the resolved model id.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DEPTH_ANYTHING_LOCAL_FILES_ONLY", "1") != "0",
        help="Use only locally cached Hugging Face files by default.",
    )
    parser.add_argument(
        "--depth-mode",
        choices=("auto", "metric", "relative"),
        default="auto",
        help="Metric models are evaluated directly; relative models are scale/shift aligned.",
    )
    parser.add_argument(
        "--align-mode",
        choices=("scale_shift", "median"),
        default="scale_shift",
        help="Alignment mode used when --depth-mode resolves to relative.",
    )
    parser.add_argument("--image-height", type=int, default=518)
    parser.add_argument("--image-width", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.9)
    parser.add_argument(
        "--topologies",
        nargs="*",
        default=None,
        help="Optional validation topologies to evaluate, e.g. topology2 topology3.",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=("auto",),
        help=(
            "Decoder/head module names for gradient extraction. Use 'auto' to "
            "select the last non-output Conv2d decoder/head layers."
        ),
    )
    parser.add_argument(
        "--num-auto-layers",
        type=int,
        default=4,
        help="Number of layers selected when --layers auto is used.",
    )
    parser.add_argument(
        "--channel-reduction",
        choices=("max_abs", "norm", "mean_abs", "sum_abs", "max"),
        default="max_abs",
        help="Channel reduction for feature-map gradients. max_abs follows the paper's max-pooling intent.",
    )
    parser.add_argument(
        "--layer-reduction",
        choices=("max", "mean", "std"),
        default="max",
        help="Reduction across multiple layer uncertainty maps. max follows the TPAMI multi-layer variant.",
    )
    parser.add_argument(
        "--reference-transform",
        choices=("flip", "gray", "noise"),
        default="flip",
        help="Image-space transform used to generate the reference depth.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Noise std for --reference-transform noise, in processor-normalized tensor space.",
    )
    parser.add_argument(
        "--prediction-activation",
        choices=("clamp", "softplus", "none"),
        default="clamp",
        help="Positive-depth stabilization applied to model predictions.",
    )
    parser.add_argument(
        "--pixels-per-batch",
        type=int,
        default=4096,
        help="Uniformly sampled valid pixels stored per batch for global pixel correlation; <=0 stores all.",
    )
    parser.add_argument(
        "--max-pixel-samples",
        type=int,
        default=100_000,
        help="Final maximum number of pixel samples used in correlation; <=0 disables the cap.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch seed used for noise reference generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA autocast for forward/backward.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print decoder/head Conv2d layer candidates and exit.",
    )
    return parser.parse_args()


def resolve_model_id(model_id: str) -> str:
    return MODEL_IDS.get(model_id, model_id)


def resolve_depth_mode(depth_mode: str, model_id: str) -> str:
    if depth_mode != "auto":
        return depth_mode
    normalized = model_id.lower()
    return "metric" if "metric" in normalized else "relative"


def processor_source(model_id: str, processor_dir: Path | None) -> str:
    if processor_dir is not None:
        return str(processor_dir)
    return model_id


def decoder_layer_candidates(model: nn.Module) -> list[str]:
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            continue
        if not (name.startswith("neck.") or name.startswith("head.")):
            continue
        if getattr(module, "out_channels", None) == 1:
            continue
        candidates.append(name)
    return candidates


def resolve_layer_names(
    model: nn.Module,
    requested_layers: Sequence[str],
    num_auto_layers: int,
) -> list[str]:
    if len(requested_layers) == 1 and requested_layers[0] == "auto":
        candidates = decoder_layer_candidates(model)
        if not candidates:
            raise ValueError("No decoder/head Conv2d candidates were found.")
        return candidates[-max(1, num_auto_layers):]
    return list(requested_layers)


def uniform_indices(
    num_items: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    if num_samples <= 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    indices = torch.arange(num_samples, device=device, dtype=torch.long)
    return indices * (num_items - 1) // (num_samples - 1)


def minmax_normalize_per_image(
    values: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    flat = values.flatten(1)
    min_values = flat.amin(dim=1).view(-1, 1, 1, 1)
    max_values = flat.amax(dim=1).view(-1, 1, 1, 1)
    denom = (max_values - min_values).clamp_min(eps)
    normalized = (values - min_values) / denom
    constant = (max_values - min_values) <= eps
    return torch.where(constant, torch.zeros_like(normalized), normalized)


def masked_image_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    values = ensure_bchw(values).float()
    mask = ensure_bchw(valid_mask).bool()
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    mask = mask & torch.isfinite(values)
    counts = mask.flatten(1).sum(dim=1)
    sums = torch.where(mask, values, torch.zeros_like(values)).flatten(1).sum(dim=1)
    means = sums / counts.clamp_min(1).to(dtype=values.dtype)
    return torch.where(counts > 0, means, means.new_full(means.shape, float("nan")))


def finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().item())


def finite_std(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() < 2:
        return float("nan")
    return float(finite.std(unbiased=False).item())


def prediction_depth(
    model: nn.Module,
    pixel_values: torch.Tensor,
    target_size: tuple[int, int],
    activation: str,
    min_depth: float,
) -> torch.Tensor:
    outputs = model(pixel_values=pixel_values)
    predicted_depth = outputs.predicted_depth
    if predicted_depth.ndim == 3:
        predicted_depth = predicted_depth.unsqueeze(1)
    predicted_depth = F.interpolate(
        predicted_depth,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )
    if activation == "clamp":
        return predicted_depth.clamp_min(min_depth)
    if activation == "softplus":
        return F.softplus(predicted_depth)
    if activation == "none":
        return predicted_depth
    raise ValueError(f"Unknown prediction activation: {activation}")


def transformed_reference_pixels(
    pixel_values: torch.Tensor,
    transform: str,
    noise_std: float,
) -> torch.Tensor:
    if transform == "flip":
        return torch.flip(pixel_values, dims=(3,))
    if transform == "gray":
        return pixel_values.mean(dim=1, keepdim=True).expand_as(pixel_values)
    if transform == "noise":
        return pixel_values + torch.randn_like(pixel_values) * noise_std
    raise ValueError(f"Unknown reference transform: {transform}")


def invert_reference_depth(
    reference_depth: torch.Tensor,
    transform: str,
) -> torch.Tensor:
    if transform == "flip":
        return torch.flip(reference_depth, dims=(3,))
    if transform in {"gray", "noise"}:
        return reference_depth
    raise ValueError(f"Unknown reference transform: {transform}")


@torch.no_grad()
def reference_depth(
    model: nn.Module,
    pixel_values: torch.Tensor,
    target_size: tuple[int, int],
    transform: str,
    noise_std: float,
    activation: str,
    min_depth: float,
) -> torch.Tensor:
    reference_pixels = transformed_reference_pixels(
        pixel_values,
        transform=transform,
        noise_std=noise_std,
    )
    depth = prediction_depth(
        model,
        reference_pixels,
        target_size=target_size,
        activation=activation,
        min_depth=min_depth,
    )
    return invert_reference_depth(depth, transform=transform)


def evaluation_depth(
    predicted_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_mode: str,
    align_mode: str,
    min_depth: float,
) -> torch.Tensor:
    predicted_depth = predicted_depth.detach()
    if depth_mode == "metric":
        return predicted_depth.clamp_min(min_depth)
    if depth_mode == "relative":
        aligned = align_relative_prediction_to_depth_space(
            predicted_depth,
            gt_depth,
            valid_mask,
            align_mode=align_mode,
        )
        return aligned["depth"]
    raise ValueError(f"Unknown depth_mode: {depth_mode}")


def evaluate_batch(
    model: nn.Module,
    extractor: GradientUncertaintyExtractor,
    batch,
    device: torch.device,
    amp: bool,
    args: argparse.Namespace,
    depth_mode: str,
    accumulator: CorrelationAccumulator,
) -> Mapping[str, torch.Tensor | float]:
    pixel_values, depth, valid_mask, _condition, _condition_stats = batch
    pixel_values = pixel_values.to(device, non_blocking=True)
    depth = depth.to(device, non_blocking=True)
    valid_mask = valid_mask.to(device, non_blocking=True)
    target_size = tuple(depth.shape[-2:])

    model.zero_grad(set_to_none=True)
    extractor.clear()
    grad_pixels = pixel_values.detach().clone().requires_grad_(True)

    with torch.autocast(device_type=device.type, enabled=amp):
        ref_depth = reference_depth(
            model=model,
            pixel_values=pixel_values.detach(),
            target_size=target_size,
            transform=args.reference_transform,
            noise_std=args.noise_std,
            activation=args.prediction_activation,
            min_depth=args.min_depth,
        )
        predicted_depth = prediction_depth(
            model=model,
            pixel_values=grad_pixels,
            target_size=target_size,
            activation=args.prediction_activation,
            min_depth=args.min_depth,
        )
        auxiliary_loss = (predicted_depth - ref_depth).square().mean()

    auxiliary_loss.backward()
    uncertainty = extractor.uncertainty_map(target_size).detach()

    pred_for_metrics = evaluation_depth(
        predicted_depth=predicted_depth,
        gt_depth=ensure_bchw(depth).float(),
        valid_mask=ensure_bchw(valid_mask).float(),
        depth_mode=depth_mode,
        align_mode=args.align_mode,
        min_depth=args.min_depth,
    )
    metrics = compute_comprehensive_depth_metrics(
        mu=pred_for_metrics.float(),
        target=ensure_bchw(depth).float(),
        valid_mask=ensure_bchw(valid_mask).float(),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    abs_rel_error, _ = depth_error_maps(
        pred_for_metrics.float(),
        ensure_bchw(depth).float(),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    valid = ensure_bchw(valid_mask).bool()
    finite_pixel_mask = (
        valid
        & torch.isfinite(abs_rel_error)
        & torch.isfinite(uncertainty)
    )

    image_uncertainty = masked_image_mean(uncertainty, finite_pixel_mask)
    accumulator.extend_images(
        uncertainty=image_uncertainty,
        abs_rel=metrics["abs_rel"],
    )
    pixels_per_batch = None if args.pixels_per_batch <= 0 else args.pixels_per_batch
    accumulator.extend_pixels(
        uncertainty=uncertainty,
        abs_rel_error=abs_rel_error,
        valid_mask=finite_pixel_mask,
        max_pixels=pixels_per_batch,
    )

    return {
        "auxiliary_loss": float(auxiliary_loss.detach().cpu().item()),
        "image_uncertainty": image_uncertainty.detach().cpu(),
        "abs_rel": metrics["abs_rel"].detach().cpu(),
    }


def correlation_row(
    level: str,
    uncertainty: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    resolved_model_id: str,
    depth_mode: str,
    layer_names: Sequence[str],
    max_samples: int | None,
) -> dict[str, object]:
    valid = torch.isfinite(uncertainty) & torch.isfinite(target)
    correlations = compute_vector_masked_correlations(
        uncertainty,
        target,
        valid_mask=valid,
        max_samples=max_samples,
        prefix="correlation",
    )
    return {
        "level": level,
        "score": "gradient_uncertainty",
        "target": "abs_rel" if level == "image" else "pixel_abs_rel_error",
        "pearson": correlations["correlation_pearson"],
        "spearman": correlations["correlation_spearman"],
        "num_samples": int(valid.sum().item()),
        "uncertainty_mean": finite_mean(uncertainty),
        "uncertainty_std": finite_std(uncertainty),
        "target_mean": finite_mean(target),
        "target_std": finite_std(target),
        "model_id": resolved_model_id,
        "depth_mode": depth_mode,
        "alignment_mode": args.align_mode if depth_mode == "relative" else "",
        "reference_transform": args.reference_transform,
        "auxiliary_loss": "mean((pred_depth - reference_depth)^2)",
        "layers": " ".join(layer_names),
        "channel_reduction": args.channel_reduction,
        "layer_reduction": args.layer_reduction if len(layer_names) > 1 else "",
        "prediction_activation": args.prediction_activation,
        "dataset_root": str(args.dataset_root),
        "topologies": "" if args.topologies is None else " ".join(args.topologies),
        "image_height": args.image_height,
        "image_width": args.image_width,
        "batch_size": args.batch_size,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
    }


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "level",
        "score",
        "target",
        "pearson",
        "spearman",
        "num_samples",
        "uncertainty_mean",
        "uncertainty_std",
        "target_mean",
        "target_std",
        "model_id",
        "depth_mode",
        "alignment_mode",
        "reference_transform",
        "auxiliary_loss",
        "layers",
        "channel_reduction",
        "layer_reduction",
        "prediction_activation",
        "dataset_root",
        "topologies",
        "image_height",
        "image_width",
        "batch_size",
        "min_depth",
        "max_depth",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    resolved_model_id = resolve_model_id(args.model_id)
    depth_mode = resolve_depth_mode(args.depth_mode, resolved_model_id)

    processor = AutoImageProcessor.from_pretrained(
        processor_source(resolved_model_id, args.processor_dir),
        cache_dir=None if args.hf_cache_dir is None else str(args.hf_cache_dir),
        local_files_only=args.local_files_only,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        resolved_model_id,
        cache_dir=None if args.hf_cache_dir is None else str(args.hf_cache_dir),
        local_files_only=args.local_files_only,
    )
    model = model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if args.list_layers:
        print("\n".join(decoder_layer_candidates(model)))
        return

    layer_names = resolve_layer_names(
        model,
        requested_layers=args.layers,
        num_auto_layers=args.num_auto_layers,
    )
    print(f"Using model: {resolved_model_id}")
    print(f"Depth mode: {depth_mode}")
    print(f"Gradient layers: {layer_names}")

    dataset = ATIRealWorldUncertaintyValidationDataset(
        root_dir=str(args.dataset_root),
        image_processor=processor,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        light_levels=LIGHT_LEVELS,
        speed_levels=MOTION_LEVELS,
        topologies=args.topologies,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=ati_collate_fn,
    )

    extractor = GradientUncertaintyExtractor(
        model=model,
        layer_names=layer_names,
        channel_reduction=args.channel_reduction,
        layer_reduction=args.layer_reduction,
    )
    accumulator = CorrelationAccumulator()

    try:
        progress = tqdm(loader, desc="Gradient uncertainty", dynamic_ncols=True)
        for step, batch in enumerate(progress, start=1):
            if batch is None:
                continue
            batch_values = evaluate_batch(
                model=model,
                extractor=extractor,
                batch=batch,
                device=device,
                amp=amp,
                args=args,
                depth_mode=depth_mode,
                accumulator=accumulator,
            )
            progress.set_postfix(
                abs_rel=f"{finite_mean(batch_values['abs_rel']):.4f}",
                unc=f"{finite_mean(batch_values['image_uncertainty']):.4f}",
                aux=f"{batch_values['auxiliary_loss']:.4e}",
            )
            if args.max_batches is not None and step >= args.max_batches:
                break
    finally:
        extractor.remove()

    image_uncertainty = accumulator.tensor("image_uncertainty")
    image_abs_rel = accumulator.tensor("image_abs_rel")
    pixel_uncertainty = accumulator.tensor("pixel_uncertainty")
    pixel_abs_rel = accumulator.tensor("pixel_abs_rel")

    max_pixel_samples = None if args.max_pixel_samples <= 0 else args.max_pixel_samples
    rows = [
        correlation_row(
            level="image",
            uncertainty=image_uncertainty,
            target=image_abs_rel,
            args=args,
            resolved_model_id=resolved_model_id,
            depth_mode=depth_mode,
            layer_names=layer_names,
            max_samples=None,
        ),
        correlation_row(
            level="pixel",
            uncertainty=pixel_uncertainty,
            target=pixel_abs_rel,
            args=args,
            resolved_model_id=resolved_model_id,
            depth_mode=depth_mode,
            layer_names=layer_names,
            max_samples=max_pixel_samples,
        ),
    ]
    write_csv(rows, args.output_csv)

    for row in rows:
        pearson = row["pearson"]
        spearman = row["spearman"]
        print(
            f"{row['level']} correlation: "
            f"pearson={pearson:.4f} spearman={spearman:.4f} "
            f"n={row['num_samples']}"
        )
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
