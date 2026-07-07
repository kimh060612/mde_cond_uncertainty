from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor


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
    ensure_bchw,
)
from model.dav2_ati_bias_model import FrozenDepthCameraGaussian  # noqa: E402
from model.dav2_ati_model import MODEL_IDS  # noqa: E402


DEFAULT_SEEN_TOPOLOGIES = ("topology2", "topology4")
DEFAULT_UNSEEN_TOPOLOGIES = ("topology3", "topology5")


@dataclass
class SplitAccumulator:
    bias: list[torch.Tensor] = field(default_factory=list)
    sigma: list[torch.Tensor] = field(default_factory=list)
    uncertainty: list[torch.Tensor] = field(default_factory=list)
    abs_rel: list[torch.Tensor] = field(default_factory=list)
    a1: list[torch.Tensor] = field(default_factory=list)

    def extend(
        self,
        sample_mask: torch.Tensor,
        bias: torch.Tensor,
        sigma: torch.Tensor,
        uncertainty: torch.Tensor,
        abs_rel: torch.Tensor,
        a1: torch.Tensor,
    ) -> None:
        sample_mask = sample_mask.detach().cpu().bool().flatten()
        if not sample_mask.any():
            return

        self.bias.append(bias.detach().cpu().float().flatten()[sample_mask])
        self.sigma.append(sigma.detach().cpu().float().flatten()[sample_mask])
        self.uncertainty.append(
            uncertainty.detach().cpu().float().flatten()[sample_mask]
        )
        self.abs_rel.append(abs_rel.detach().cpu().float().flatten()[sample_mask])
        self.a1.append(a1.detach().cpu().float().flatten()[sample_mask])

    def tensor(self, name: str) -> torch.Tensor:
        values = getattr(self, name)
        if not values:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(values, dim=0).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate FrozenDepthCameraGaussian bias/sigma components with "
            "validation AbsRel and A1."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained FrozenDepthCameraGaussian checkpoint.",
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
        default=Path("evaluation_models") / "decomposed_correlation.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="small",
        help="Depth Anything V2 model alias from MODEL_IDS or a full HF model id.",
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        default=None,
        help="Optional image processor directory. Defaults to checkpoint dir if available.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument("--image-height", type=int, default=518)
    parser.add_argument("--image-width", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.9)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--film-hidden-dim", type=int, default=None)
    parser.add_argument("--max-bias", type=float, default=0.5)
    parser.add_argument("--min-log-var", type=float, default=-2.5)
    parser.add_argument("--max-log-var", type=float, default=1.0)
    parser.add_argument("--initial-std", type=float, default=0.5)
    parser.add_argument("--variance-head-init-std", type=float, default=1e-3)
    parser.add_argument(
        "--seen-topologies",
        nargs="*",
        default=None,
        help="Seen validation topologies. Defaults to checkpoint metadata or topology2/topology4.",
    )
    parser.add_argument(
        "--unseen-topologies",
        nargs="*",
        default=None,
        help="Unseen validation topologies. Defaults to checkpoint metadata or topology3/topology5.",
    )
    parser.add_argument(
        "--prediction-source",
        choices=("aligned_corrected", "aligned_base"),
        default="aligned_base",
        help=(
            "Depth map used for AbsRel/A1. Both choices always apply "
            "scale/shift alignment to base depth first."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA autocast.",
    )
    return parser.parse_args()


def resolve_model_id(model_id: str) -> str:
    return MODEL_IDS.get(model_id, model_id)


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, Mapping):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        metadata = checkpoint.get("dataset_metadata", {}) or {}
    else:
        state_dict = checkpoint
        metadata = {}

    if state_dict and all(
        isinstance(key, str) and key.startswith("module.")
        for key in state_dict
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return checkpoint, state_dict, metadata


def infer_hidden_channels(state_dict: Mapping[str, torch.Tensor], fallback: int) -> int:
    weight = state_dict.get("feature_projection.0.weight")
    if weight is None:
        return fallback
    return int(weight.shape[0])


def infer_film_hidden_dim(state_dict: Mapping[str, torch.Tensor], fallback: int) -> int:
    weight = state_dict.get("film_generator.1.weight")
    if weight is None:
        return fallback
    return int(weight.shape[0])


def validate_bias_checkpoint(state_dict: Mapping[str, torch.Tensor]) -> None:
    has_bias = any(key.startswith("bias_head.") for key in state_dict)
    has_variance = any(key.startswith("variance_head.") for key in state_dict)
    if not has_bias or not has_variance:
        raise ValueError(
            "This checkpoint does not look like FrozenDepthCameraGaussian: "
            "missing bias_head and/or variance_head weights."
        )


def processor_source(
    checkpoint_path: Path,
    model_id: str,
    processor_dir: Path | None,
) -> str:
    if processor_dir is not None:
        return str(processor_dir)
    checkpoint_dir = checkpoint_path.resolve().parent
    if (checkpoint_dir / "preprocessor_config.json").is_file():
        return str(checkpoint_dir)
    return model_id


def metadata_sequence(
    metadata: Mapping,
    key: str,
    fallback: Sequence[str],
) -> list[str]:
    value = metadata.get(key)
    if value:
        return [str(item) for item in value]
    return list(fallback)


def topology_number(topology: str) -> int:
    value = str(topology).strip()
    if value.startswith("topology"):
        value = value[len("topology") :]
    if not value.isdigit():
        raise ValueError(f"Expected topology or integer, got {topology}")
    return int(value)


def topology_numbers(values: Sequence[str]) -> torch.Tensor:
    return torch.tensor(
        [topology_number(value) for value in values],
        dtype=torch.long,
    )


def apply_metadata_normalization(dataset, metadata: Mapping) -> None:
    for attr in ("exposure_min", "exposure_max", "gain_min", "gain_max"):
        value = metadata.get(attr)
        if value is not None:
            setattr(dataset, attr, float(value))


def build_model(
    args: argparse.Namespace,
    model_id: str,
    context_dim: int,
    state_dict: Mapping[str, torch.Tensor],
) -> FrozenDepthCameraGaussian:
    hidden_channels = (
        infer_hidden_channels(state_dict, 64)
        if args.hidden_channels is None
        else int(args.hidden_channels)
    )
    film_hidden_dim = (
        infer_film_hidden_dim(state_dict, 128)
        if args.film_hidden_dim is None
        else int(args.film_hidden_dim)
    )
    return FrozenDepthCameraGaussian(
        model_id=model_id,
        context_dim=context_dim,
        cache_dir=None if args.hf_cache_dir is None else str(args.hf_cache_dir),
        feature_channels=hidden_channels,
        hidden_channels=hidden_channels,
        film_hidden_dim=film_hidden_dim,
        max_bias=args.max_bias,
        min_log_variance=args.min_log_var,
        max_log_variance=args.max_log_var,
        initial_std=args.initial_std,
        variance_head_init_std=args.variance_head_init_std,
    )


def masked_image_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    values = ensure_bchw(values).float()
    mask = ensure_bchw(valid_mask).bool()
    if mask.shape != values.shape:
        mask = mask.expand_as(values)

    counts = mask.flatten(1).sum(dim=1)
    sums = (
        torch.where(mask, values, torch.zeros_like(values))
        .flatten(1)
        .sum(dim=1)
    )
    means = sums / counts.clamp_min(1).to(dtype=values.dtype)
    return torch.where(
        counts > 0,
        means,
        means.new_full(means.shape, float("nan")),
    )


def prediction_from_output(
    output: Mapping[str, torch.Tensor],
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    prediction_source: str,
) -> torch.Tensor:
    aligned = align_relative_prediction_to_depth_space(
        output["base_depth"],
        depth,
        valid_mask,
        align_mode="scale_shift",
    )
    if prediction_source == "aligned_base":
        return aligned["depth"]
    if prediction_source == "aligned_corrected":
        return aligned["depth"] + ensure_bchw(output["camera_bias"]).float()
    raise ValueError(f"Unknown prediction_source: {prediction_source}")


@torch.no_grad()
def evaluate_batch(
    model: FrozenDepthCameraGaussian,
    batch,
    device: torch.device,
    amp: bool,
    prediction_source: str,
    min_depth: float,
    max_depth: float,
) -> dict[str, torch.Tensor]:
    pixel_values, depth, valid_mask, condition, condition_stats = batch
    pixel_values = pixel_values.to(device, non_blocking=True)
    depth = depth.to(device, non_blocking=True)
    valid_mask = valid_mask.to(device, non_blocking=True)
    condition = condition.to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, enabled=amp):
        output = model(
            pixel_values,
            context=condition,
            target_size=depth.shape[-2:],
        )

    pred_depth = prediction_from_output(
        output=output,
        depth=depth.float(),
        valid_mask=valid_mask,
        prediction_source=prediction_source,
    )
    metrics = compute_comprehensive_depth_metrics(
        mu=pred_depth.float(),
        target=depth.float(),
        valid_mask=valid_mask.float(),
        min_depth=min_depth,
        max_depth=max_depth,
    )
    bias_score = masked_image_mean(
        ensure_bchw(output["camera_bias"]).detach().float().abs(),
        valid_mask,
    )
    sigma_score = masked_image_mean(
        ensure_bchw(output["std"]).detach().float(),
        valid_mask,
    )
    uncertainty_score = masked_image_mean(
        torch.sqrt(
            ensure_bchw(output["camera_bias"]).detach().float().square()
            + ensure_bchw(output["std"]).detach().float().square()
        ),
        valid_mask,
    )
    return {
        "bias": bias_score.detach().cpu(),
        "sigma": sigma_score.detach().cpu(),
        "uncertainty": uncertainty_score.detach().cpu(),
        "abs_rel": metrics["abs_rel"].detach().cpu(),
        "a1": metrics["a1"].detach().cpu(),
        "topology": condition_stats[:, 6].detach().cpu().long(),
    }


def extend_splits(
    split_accumulators: dict[str, SplitAccumulator],
    batch_values: Mapping[str, torch.Tensor],
    seen_numbers: torch.Tensor,
    unseen_numbers: torch.Tensor,
) -> None:
    topology = batch_values["topology"].long()
    total_mask = torch.ones_like(topology, dtype=torch.bool)
    seen_mask = torch.isin(topology, seen_numbers)
    unseen_mask = torch.isin(topology, unseen_numbers)

    for split_name, sample_mask in (
        ("total", total_mask),
        ("seen", seen_mask),
        ("unseen", unseen_mask),
    ):
        split_accumulators[split_name].extend(
            sample_mask=sample_mask,
            bias=batch_values["bias"],
            sigma=batch_values["sigma"],
            uncertainty=batch_values["uncertainty"],
            abs_rel=batch_values["abs_rel"],
            a1=batch_values["a1"],
        )


def finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().item())


def correlation_rows(
    split_accumulators: Mapping[str, SplitAccumulator],
    args: argparse.Namespace,
    model_id: str,
    checkpoint_metadata: Mapping,
    seen_topologies: Sequence[str],
    unseen_topologies: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    components = {
        "bias": ("mean_abs_camera_bias", "mean(abs(b_c))"),
        "sigma": ("mean_sigma", "mean(sigma)"),
        "uncertainty": (
            "mean_sqrt_bias2_sigma2",
            "mean(sqrt(b_c^2 + sigma^2))",
        ),
    }
    targets = {
        "abs_rel": "abs_rel",
        "a1": "a1",
    }
    for split_name in ("total", "seen", "unseen"):
        accumulator = split_accumulators[split_name]
        for component_key, (component_name, component_formula) in components.items():
            component_values = accumulator.tensor(component_key)
            for target_key, target_name in targets.items():
                target_values = accumulator.tensor(target_key)
                valid_mask = (
                    torch.isfinite(component_values)
                    & torch.isfinite(target_values)
                )
                correlations = compute_vector_masked_correlations(
                    component_values,
                    target_values,
                    valid_mask=valid_mask,
                    max_samples=None,
                    prefix="correlation",
                )
                rows.append(
                    {
                        "checkpoint": str(args.checkpoint),
                        "epoch": checkpoint_metadata.get("epoch", ""),
                        "model_id": model_id,
                        "prediction_source": args.prediction_source,
                        "alignment_mode": "scale_shift",
                        "split": split_name,
                        "seen_topologies": " ".join(seen_topologies),
                        "unseen_topologies": " ".join(unseen_topologies),
                        "component": component_name,
                        "component_formula": component_formula,
                        "target_metric": target_name,
                        "num_samples": int(valid_mask.sum().item()),
                        "component_mean": finite_mean(component_values),
                        "target_mean": finite_mean(target_values),
                        "pearson": correlations["correlation_pearson"],
                        "spearman": correlations["correlation_spearman"],
                    }
                )
    return rows


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "epoch",
        "model_id",
        "prediction_source",
        "alignment_mode",
        "split",
        "seen_topologies",
        "unseen_topologies",
        "component",
        "component_formula",
        "target_metric",
        "num_samples",
        "component_mean",
        "target_mean",
        "pearson",
        "spearman",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp

    checkpoint, state_dict, metadata = load_checkpoint(args.checkpoint)
    validate_bias_checkpoint(state_dict)
    model_id = resolve_model_id(args.model_id)

    seen_topologies = (
        args.seen_topologies
        if args.seen_topologies is not None
        else metadata_sequence(
            metadata,
            "seen_validation_topologies",
            DEFAULT_SEEN_TOPOLOGIES,
        )
    )
    unseen_topologies = (
        args.unseen_topologies
        if args.unseen_topologies is not None
        else metadata_sequence(
            metadata,
            "unseen_validation_topologies",
            DEFAULT_UNSEEN_TOPOLOGIES,
        )
    )
    seen_numbers = topology_numbers(seen_topologies)
    unseen_numbers = topology_numbers(unseen_topologies)
    print(f"Seen topologies: {seen_topologies} -> {seen_numbers.tolist()}")
    print(f"Unseen topologies: {unseen_topologies} -> {unseen_numbers.tolist()}")
    
    image_processor = AutoImageProcessor.from_pretrained(
        processor_source(args.checkpoint, model_id, args.processor_dir),
        cache_dir=None if args.hf_cache_dir is None else str(args.hf_cache_dir),
    )
    dataset = ATIRealWorldUncertaintyValidationDataset(
        root_dir=str(args.dataset_root),
        image_processor=image_processor,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        light_levels=LIGHT_LEVELS,
        speed_levels=MOTION_LEVELS,
        topologies=seen_topologies + unseen_topologies,
    )
    apply_metadata_normalization(dataset, metadata)

    model = build_model(
        args=args,
        model_id=model_id,
        context_dim=dataset.condition_dim,
        state_dict=state_dict,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=ati_collate_fn,
    )
    split_accumulators = {
        "total": SplitAccumulator(),
        "seen": SplitAccumulator(),
        "unseen": SplitAccumulator(),
    }

    progress = tqdm(loader, desc="Bias/sigma correlation", dynamic_ncols=True)
    for step, batch in enumerate(progress, start=1):
        if batch is None:
            continue
        batch_values = evaluate_batch(
            model=model,
            batch=batch,
            device=device,
            amp=amp,
            prediction_source=args.prediction_source,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        extend_splits(
            split_accumulators=split_accumulators,
            batch_values=batch_values,
            seen_numbers=seen_numbers,
            unseen_numbers=unseen_numbers,
        )
        progress.set_postfix(
            abs_rel=f"{finite_mean(batch_values['abs_rel']):.4f}",
            a1=f"{finite_mean(batch_values['a1']):.4f}",
        )
        if args.max_batches is not None and step >= args.max_batches:
            break

    checkpoint_info = checkpoint if isinstance(checkpoint, Mapping) else {}
    rows = correlation_rows(
        split_accumulators=split_accumulators,
        args=args,
        model_id=model_id,
        checkpoint_metadata=checkpoint_info,
        seen_topologies=seen_topologies,
        unseen_topologies=unseen_topologies,
    )
    write_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
