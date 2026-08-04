from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.ati_dataset_caminduce import (  # noqa: E402
    CameraParameterRange,
    FoundationCameraGroupedDataset,
    PairedResizeToTensor,
)
from evaluation_utils.eval_metrics import compute_vector_masked_correlations  # noqa: E402
from evaluation_utils.eval_selection import (  # noqa: E402
    compute_selection_metrics,
    compute_selection_metrics_rank_topk,
)
from model.dav2_model import MODEL_IDS  # noqa: E402
from model.loss_fn import (  # noqa: E402
    log_scale_invariant_depth_difference,
)
from model.loss_target import (  # noqa: E402
    ordinal_structure_failure,
    ssi_independent_depth_loss,
    ssi_independent_meter_space_depth_loss,
    ssi_depth_guided_meter_space_depth_loss
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "base_caminduce.yaml"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "loss_performance_correlation.csv"
DEFAULT_DATASET_ROOT = Path("/datasets/ATI/MDE/orbbec_realworld_dataset")
DEFAULT_REPLACEABLE_DATASET_PREFIXES = (
    "/dataset/ATI/MDE/orbbec_realworld_dataset",
    "/datasets/ATI/MDE/orbbec_realworld_dataset",
    "/media/michael/ssd1/AIoT_ATI/orbbec_realworld_dataset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict candidate/canonical RGB pairs with Depth Anything V2 and "
            "save correlations between prediction-space losses and AbsRel "
            "performance degradation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Config YAML used for dataset/model defaults.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="CSV file or directory containing matched camera-induced pairs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root used to remap RGB paths stored in the CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to save the summary correlation CSV.",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default=None,
        help=(
            "Depth Anything V2 key or Hugging Face model id. "
            "Defaults to config model.model_id."
        ),
    )
    parser.add_argument(
        "--foundation-model-name",
        type=str,
        default=None,
        help="Foundation model name expected by FoundationCameraGroupedDataset.",
    )
    parser.add_argument(
        "--camera-model-name",
        type=str,
        default=None,
        help="Physical camera model name expected by FoundationCameraGroupedDataset.",
    )
    parser.add_argument(
        "--topologies",
        nargs="*",
        default=["topology1", "topology2", "topology3", "topology4", "topology5"],
        help="Optional topology filter. Use no values after the flag for all topologies.",
    )
    parser.add_argument(
        "--candidates-per-group",
        type=int,
        default=None,
        help="Candidates per canonical group. Defaults to config training.candidates_per_group.",
    )
    parser.add_argument(
        "--candidate-sampling",
        choices=("random", "parameter_diverse"),
        default="parameter_diverse",
        help="Sampling mode passed to FoundationCameraGroupedDataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader batch size in canonical groups.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=32,
        help="Depth Anything V2 forward batch size after flattening candidates.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults to config dataset.num_workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Dataset sampling seed. Defaults to config training.seed.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="Dataset min depth metadata. Defaults to config dataset.min_depth.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Dataset max depth metadata. Defaults to config dataset.max_depth.",
    )
    parser.add_argument(
        "--path-replacement",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Additional RGB path prefix replacement. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Depth Anything V2 only from local Hugging Face cache.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for Depth Anything V2 inference.",
    )
    parser.add_argument(
        "--correlation-max-samples",
        type=int,
        default=100_000,
        help="Maximum number of pair-level samples used by the correlation helper.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional debug limit for canonical groups.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit for DataLoader batches.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on the first batch error instead of skipping it.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Any:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        from omegaconf import OmegaConf

        return OmegaConf.load(config_path)
    except ModuleNotFoundError:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install omegaconf or PyYAML to read the YAML config."
            ) from exc

        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)


def cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    value: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(value, dict):
            value = value.get(part, default)
        else:
            value = getattr(value, part, default)
        if value is default:
            return default
    return value


def resolve_csv_paths(csv_path: Path) -> list[Path]:
    if csv_path.is_file():
        return [csv_path]
    if csv_path.is_dir():
        paths = sorted(csv_path.glob("*.csv"))
        if paths:
            return paths
    raise FileNotFoundError(f"No CSV files found at {csv_path}")


def resolve_model_id(model_name_or_id: str) -> str:
    return MODEL_IDS.get(model_name_or_id, model_name_or_id)


def parse_topologies(raw_topologies: list[str] | None) -> list[str] | None:
    if raw_topologies == []:
        return None
    return raw_topologies


def parse_path_replacements(
    replacements: Iterable[str],
    dataset_root: Path | None,
) -> dict[str, str]:
    result: dict[str, str] = {}

    if dataset_root is not None:
        for old_prefix in DEFAULT_REPLACEABLE_DATASET_PREFIXES:
            result[old_prefix] = str(dataset_root)

    for replacement in replacements:
        if "=" not in replacement:
            raise ValueError(
                f"Invalid --path-replacement '{replacement}'. Expected OLD=NEW."
            )
        old, new = replacement.split("=", 1)
        if not old:
            raise ValueError("Path replacement OLD prefix must not be empty.")
        result[old] = new

    return result


def build_dataset(
    args: argparse.Namespace,
    cfg: Any,
    processor: AutoImageProcessor,
) -> FoundationCameraGroupedDataset:
    csv_path = args.csv_path or Path(str(cfg_get(cfg, "dataset.csv_path")))
    min_depth = float(args.min_depth or cfg_get(cfg, "dataset.min_depth", 1e-3))
    max_depth = float(args.max_depth or cfg_get(cfg, "dataset.max_depth", 10.0))
    candidates_per_group = int(
        args.candidates_per_group
        or cfg_get(cfg, "training.candidates_per_group", 4)
    )

    foundation_model_name = (
        args.foundation_model_name
        or str(cfg_get(cfg, "model.model_id"))
    )
    camera_model_name = (
        args.camera_model_name
        or str(cfg_get(cfg, "model.camera_model_name"))
    )

    return FoundationCameraGroupedDataset(
        csv_paths=resolve_csv_paths(csv_path),
        foundation_model_name=foundation_model_name,
        camera_model_name=camera_model_name,
        parameter_range=CameraParameterRange(
            exposure_min=float(cfg_get(cfg, "dataset.exposure_min")),
            exposure_max=float(cfg_get(cfg, "dataset.exposure_max")),
            gain_min=float(cfg_get(cfg, "dataset.gain_min")),
            gain_max=float(cfg_get(cfg, "dataset.gain_max")),
        ),
        candidates_per_group=max(2, candidates_per_group),
        candidate_sampling=args.candidate_sampling,
        parameter_normalization="linear",
        context_output_range="zero_one",
        path_replacements=parse_path_replacements(
            args.path_replacement,
            args.dataset_root,
        ),
        pair_transform=PairedResizeToTensor(image_processor=processor),
        topologies=parse_topologies(args.topologies),
        load_images=True,
        load_depth=True,
        min_depth=min_depth,
        max_depth=max_depth,
        seed=int(args.seed or cfg_get(cfg, "training.seed", 42)),
    )


def maybe_subset_dataset(
    dataset: FoundationCameraGroupedDataset,
    max_groups: int | None,
) -> FoundationCameraGroupedDataset | Subset:
    if max_groups is None:
        return dataset
    return Subset(dataset, range(min(max_groups, len(dataset))))


@torch.inference_mode()
def predict_depth(
    model: AutoModelForDepthEstimation,
    processor: AutoImageProcessor,
    pixel_values: torch.Tensor,
    *,
    target_size: tuple[int, int],
    inference_batch_size: int,
) -> torch.Tensor:
    depths: list[torch.Tensor] = []
    batch_size = max(1, int(inference_batch_size))

    for start in range(0, pixel_values.shape[0], batch_size):
        chunk = pixel_values[start : start + batch_size]
        outputs = model(pixel_values=chunk)
        processed = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[target_size] * chunk.shape[0],
        )
        depths.extend(entry["predicted_depth"].float() for entry in processed)

    return torch.stack(depths, dim=0).unsqueeze(1)


def finite_stats(values: torch.Tensor) -> tuple[float, float, float, float]:
    values = values.detach().float().flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(values.mean().item()),
        float(values.std(unbiased=False).item()),
        float(values.min().item()),
        float(values.max().item()),
    )


def summarize_loss_correlation(
    *,
    loss_name: str,
    loss_values: torch.Tensor,
    degradation_values: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    group_ids: torch.Tensor,
    metadata: dict[str, object],
    max_samples: int,
) -> dict[str, object]:
    valid_mask = torch.isfinite(loss_values) & torch.isfinite(degradation_values)
    prefix = f"{loss_name}_vs_abs_rel_degradation"
    correlations = compute_vector_masked_correlations(
        loss_values,
        degradation_values,
        valid_mask=valid_mask,
        max_samples=max_samples,
        prefix=prefix,
    )
    loss_mean, loss_std, loss_min, loss_max = finite_stats(loss_values[valid_mask])
    deg_mean, deg_std, deg_min, deg_max = finite_stats(degradation_values[valid_mask])
    selection_metrics = {
        **compute_selection_metrics(
            loss_values,
            candidate_abs_rel,
            group_ids,
            min_settings_per_group=2,
        ),
        **compute_selection_metrics_rank_topk(
            loss_values,
            candidate_abs_rel,
            group_ids,
            min_settings_per_group=2,
        ),
    }

    return {
        "loss_name": loss_name,
        "target_name": "performance_degradation_abs_rel",
        "pearson": correlations[f"{prefix}_pearson"],
        "spearman": correlations[f"{prefix}_spearman"],
        "num_valid_pairs": int(valid_mask.sum().item()),
        "loss_mean": loss_mean,
        "loss_std": loss_std,
        "loss_min": loss_min,
        "loss_max": loss_max,
        "abs_rel_degradation_mean": deg_mean,
        "abs_rel_degradation_std": deg_std,
        "abs_rel_degradation_min": deg_min,
        "abs_rel_degradation_max": deg_max,
        **selection_metrics,
        **metadata,
    }


@torch.inference_mode()
def collect_loss_values(
    *,
    model: AutoModelForDepthEstimation,
    processor: AutoImageProcessor,
    loader: DataLoader,
    device: torch.device,
    inference_batch_size: int,
    max_batches: int | None,
    strict: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    log_losses: list[torch.Tensor] = []
    ssi_independent_losses: list[torch.Tensor] = []
    ssi_independent_meter_space_losses: list[torch.Tensor] = []
    ssi_depth_guided_meter_space_losses: list[torch.Tensor] = []
    abs_rel_degradations: list[torch.Tensor] = []
    candidate_abs_rels: list[torch.Tensor] = []
    group_ids: list[torch.Tensor] = []
    ordinal_losses: list[torch.Tensor] = []

    counters: dict[str, object] = {
        "num_total_groups": 0,
        "num_total_pairs": 0,
        "num_processed_batches": 0,
        "num_skipped_batches": 0,
    }

    progress = tqdm(loader, desc="DA-v2 loss correlation", dynamic_ncols=True)
    for batch_index, batch in enumerate(progress, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        try:
            candidate_images = batch["candidate_images"]
            canonical_images = batch["canonical_images"]
            if candidate_images.ndim != 5:
                raise ValueError(
                    "candidate_images must have shape [G, K, C, H, W], "
                    f"got {tuple(candidate_images.shape)}"
                )

            num_groups, num_candidates = candidate_images.shape[:2]
            flat_candidates = candidate_images.reshape(
                num_groups * num_candidates,
                *candidate_images.shape[2:],
            )
            unique_canonicals = canonical_images[:, 0]

            candidate_pixel_values = flat_candidates.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            canonical_pixel_values = unique_canonicals.to(
                device=device, dtype=torch.float32, non_blocking=True
            )

            candidate_depth = predict_depth(
                model,
                processor,
                candidate_pixel_values,
                target_size=tuple(batch["candidate_depths"].shape[-2:]),
                inference_batch_size=inference_batch_size,
            )
            canonical_depth = predict_depth(
                model,
                processor,
                canonical_pixel_values,
                target_size=tuple(batch["canonical_depths"].shape[-2:]),
                inference_batch_size=inference_batch_size,
            ).repeat_interleave(num_candidates, dim=0)

            candidate_gt_depth = batch["candidate_depths"].reshape(
                num_groups * num_candidates,
                *batch["candidate_depths"].shape[2:],
            ).unsqueeze(1).to(device=device, dtype=torch.float32, non_blocking=True)
            canonical_gt_depth = batch["canonical_depths"].reshape(
                num_groups * num_candidates,
                *batch["canonical_depths"].shape[2:],
            ).unsqueeze(1).to(device=device, dtype=torch.float32, non_blocking=True)

            log_loss = log_scale_invariant_depth_difference(
                candidate_depth,
                canonical_depth,
            )
            ssi_independent_loss = ssi_independent_depth_loss(
                candidate_depth,
                canonical_depth,
                candidate_gt_depth,
                canonical_gt_depth,
            )
            ssi_independent_meter_space_loss = ssi_independent_meter_space_depth_loss(
                candidate_depth,
                canonical_depth,
                candidate_gt_depth,
                canonical_gt_depth,
            )
            ssi_depth_guided_meter_space_loss = ssi_depth_guided_meter_space_depth_loss(
                candidate_depth,
                canonical_depth,
                candidate_gt_depth,
                canonical_gt_depth,
            )
            degradation = batch["abs_rel_degradation"].reshape(-1).float()

            log_losses.append(log_loss.detach().cpu().float())
            ssi_independent_losses.append(ssi_independent_loss.detach().cpu().float())
            ssi_independent_meter_space_losses.append(
                ssi_independent_meter_space_loss.detach().cpu().float()
            )
            ssi_depth_guided_meter_space_losses.append(
                ssi_depth_guided_meter_space_loss.detach().cpu().float()
            )
            abs_rel_degradations.append(degradation.cpu())
            candidate_abs_rels.append(batch["candidate_abs_rel"].reshape(-1).cpu().float())
            group_ids.append(batch["group_index"].repeat_interleave(num_candidates).cpu())

            counters["num_total_groups"] = int(counters["num_total_groups"]) + num_groups
            counters["num_total_pairs"] = int(counters["num_total_pairs"]) + (
                num_groups * num_candidates
            )
            counters["num_processed_batches"] = int(counters["num_processed_batches"]) + 1

            progress.set_postfix(
                log=f"{finite_stats(log_loss)[0]:.4f}",
                ssi=f"{finite_stats(ssi_independent_loss)[0]:.4f}",
                meter=f"{finite_stats(ssi_independent_meter_space_loss)[0]:.4f}",
                deg=f"{finite_stats(degradation)[0]:.4f}",
            )
        except Exception:
            counters["num_skipped_batches"] = int(counters["num_skipped_batches"]) + 1
            if strict:
                raise

    if not log_losses:
        raise RuntimeError("No valid batches were evaluated. Check dataset paths and filters.")

    return (
        torch.cat(log_losses, dim=0),
        torch.cat(ssi_independent_losses, dim=0),
        torch.cat(ssi_independent_meter_space_losses, dim=0),
        torch.cat(ssi_depth_guided_meter_space_losses, dim=0),
        torch.cat(abs_rel_degradations, dim=0),
        torch.cat(candidate_abs_rels, dim=0),
        torch.cat(group_ids, dim=0),
        counters,
    )


def write_summary_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object) -> str:
    if not isinstance(value, float):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    depth_model_name = args.depth_model or str(cfg_get(cfg, "model.model_id"))
    model_id = resolve_model_id(depth_model_name)
    cache_dir = None if args.hf_cache_dir is None else str(args.hf_cache_dir)

    processor = AutoImageProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        use_fast=False,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    model = model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = build_dataset(args, cfg, processor)
    eval_dataset = maybe_subset_dataset(dataset, args.max_groups)
    num_workers = int(args.num_workers if args.num_workers is not None else cfg_get(cfg, "dataset.num_workers", 0))
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    print(f"Using Depth Anything V2 model: {model_id}")
    print(
        "Dataset groups: "
        f"{len(eval_dataset):,} / {len(dataset):,}, "
        f"batch_size={args.batch_size}, "
        f"inference_batch_size={args.inference_batch_size}"
    )

    (
        log_losses,
        ssi_independent_losses,
        ssi_independent_meter_space_losses,
        ssi_depth_guided_meter_space_losses,
        degradations,
        candidate_abs_rels,
        group_ids,
        counters,
    ) = collect_loss_values(
        model=model,
        processor=processor,
        loader=loader,
        device=device,
        inference_batch_size=args.inference_batch_size,
        max_batches=args.max_batches,
        strict=args.strict,
    )

    candidates_per_group = int(
        args.candidates_per_group
        or cfg_get(cfg, "training.candidates_per_group", 4)
    )
    metadata = {
        **counters,
        "depth_model_name": depth_model_name,
        "depth_model_id": model_id,
        "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "candidates_per_group": candidates_per_group,
        "processor_preprocessing": 1,
        "softplus_depth": 0,
    }

    rows = [
        summarize_loss_correlation(
            loss_name="log_scale_invariant_depth_difference",
            loss_values=log_losses,
            degradation_values=degradations,
            candidate_abs_rel=candidate_abs_rels,
            group_ids=group_ids,
            metadata=metadata,
            max_samples=args.correlation_max_samples,
        ),
        summarize_loss_correlation(
            loss_name="ssi_independent_depth_loss",
            loss_values=ssi_independent_losses,
            degradation_values=degradations,
            candidate_abs_rel=candidate_abs_rels,
            group_ids=group_ids,
            metadata=metadata,
            max_samples=args.correlation_max_samples,
        ),
        summarize_loss_correlation(
            loss_name="ssi_independent_meter_space_depth_loss",
            loss_values=ssi_independent_meter_space_losses,
            degradation_values=degradations,
            candidate_abs_rel=candidate_abs_rels,
            group_ids=group_ids,
            metadata=metadata,
            max_samples=args.correlation_max_samples,
        ),
        summarize_loss_correlation(
            loss_name="ssi_depth_guided_meter_space_depth_loss",
            loss_values=ssi_depth_guided_meter_space_losses,
            degradation_values=degradations,
            candidate_abs_rel=candidate_abs_rels,
            group_ids=group_ids,
            metadata=metadata,
            max_samples=args.correlation_max_samples,
        )
    ]

    write_summary_csv(rows, args.output_csv)

    print(f"Saved correlation summary to {args.output_csv}")
    for row in rows:
        print(
            f"{row['loss_name']}: "
            f"pearson={format_float(row['pearson'])}, "
            f"spearman={format_float(row['spearman'])}, "
            f"top1/3/5={format_float(row['selection_top1_accuracy'])}/"
            f"{format_float(row['selection_top3_accuracy'])}/"
            f"{format_float(row['selection_top5_accuracy'])}, "
            f"E<3/5/10%={format_float(row['selection_accuracy_within_3pct'])}/"
            f"{format_float(row['selection_accuracy_within_5pct'])}/"
            f"{format_float(row['selection_accuracy_within_10pct'])}, "
            f"regret={format_float(row['selection_mean_regret_abs_rel'])}, "
            f"n={row['num_valid_pairs']}"
        )


if __name__ == "__main__":
    main()
