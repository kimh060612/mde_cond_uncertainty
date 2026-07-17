from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.ati_dataset_caminduce import (  # noqa: E402
    CameraParameterRange,
    FoundationCameraGroupedDataset,
)
from evaluation_utils.eval_metrics import compute_vector_masked_correlations  # noqa: E402
from model.loss_fn import (  # noqa: E402
    log_scale_invariant_depth_difference,
    scale_shift_invariant_depth_loss,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "base_caminduce.yaml"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "loss_performance_correlation.csv"
DEFAULT_DATASET_PATH_PREFIX = "/dataset/ATI/MDE/orbbec_realworld_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute correlations between depth-difference losses and "
            "AbsRel performance degradation from ati_dataset_caminduce data."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Config YAML used for default dataset/model parameters.",
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
        default="/dataset/ATI/MDE/orbbec_realworld_dataset",
        help="Dataset root used to remap stored absolute paths.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to save the summary correlation CSV.",
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
        help="Optional topology filter, e.g. topology1 topology2. Default: all.",
    )
    parser.add_argument(
        "--candidates-per-group",
        type=int,
        default=2,
        help=(
            "Minimum distinct camera settings required for dataset construction. "
            "Rows are still evaluated individually."
        ),
    )
    parser.add_argument(
        "--candidate-sampling",
        choices=("random", "parameter_diverse"),
        default="parameter_diverse",
        help="Sampling mode passed to the dataset constructor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset sampling seed. Correlation uses all filtered table rows.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=1e-3,
        help="Minimum valid depth. Defaults to config dataset.min_depth.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum valid depth. Defaults to config dataset.max_depth.",
    )
    parser.add_argument(
        "--path-replacement",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Additional path prefix replacement. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for loss computation.",
    )
    parser.add_argument(
        "--correlation-max-samples",
        type=int,
        default=100_000,
        help="Maximum number of row-level samples used by the correlation helper.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug limit for evaluated rows.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on the first row error instead of skipping invalid rows.",
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


def parse_path_replacements(
    replacements: Iterable[str],
    dataset_root: Path | None,
) -> dict[str, str]:
    result: dict[str, str] = {}

    if dataset_root is not None:
        result[DEFAULT_DATASET_PATH_PREFIX] = str(dataset_root)

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


def build_dataset(args: argparse.Namespace, cfg: Any) -> FoundationCameraGroupedDataset:
    csv_path = args.csv_path or Path(str(cfg_get(cfg, "dataset.csv_path")))
    dataset_root = args.dataset_root or Path(str(cfg_get(cfg, "dataset.dataset_root")))
    min_depth = float(args.min_depth or cfg_get(cfg, "dataset.min_depth", 1e-3))
    max_depth = float(args.max_depth or cfg_get(cfg, "dataset.max_depth", 10.0))

    foundation_model_name = (
        args.foundation_model_name
        or str(cfg_get(cfg, "model.model_id"))
    )
    camera_model_name = (
        args.camera_model_name
        or str(cfg_get(cfg, "model.camera_model_name"))
    )

    csv_paths = resolve_csv_paths(csv_path)
    path_replacements = parse_path_replacements(
        args.path_replacement,
        dataset_root,
    )

    return FoundationCameraGroupedDataset(
        csv_paths=csv_paths,
        foundation_model_name=foundation_model_name,
        camera_model_name=camera_model_name,
        parameter_range=CameraParameterRange(
            exposure_min=float(cfg_get(cfg, "dataset.exposure_min")),
            exposure_max=float(cfg_get(cfg, "dataset.exposure_max")),
            gain_min=float(cfg_get(cfg, "dataset.gain_min")),
            gain_max=float(cfg_get(cfg, "dataset.gain_max")),
        ),
        candidates_per_group=max(2, int(args.candidates_per_group)),
        candidate_sampling=args.candidate_sampling,
        parameter_normalization="linear",
        context_output_range="zero_one",
        path_replacements=path_replacements,
        topologies=args.topologies,
        load_images=False,
        load_depth=False,
        min_depth=min_depth,
        max_depth=max_depth,
        seed=args.seed,
    )


def make_valid_mask(
    candidate_depth: torch.Tensor,
    canonical_depth: torch.Tensor,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    mask = torch.isfinite(candidate_depth) & torch.isfinite(canonical_depth)
    mask &= candidate_depth > min_depth
    mask &= candidate_depth < max_depth
    mask &= canonical_depth > min_depth
    mask &= canonical_depth < max_depth
    return mask


def scalar_from_loss(loss: torch.Tensor) -> float:
    loss = loss.detach().flatten()
    if loss.numel() != 1:
        raise ValueError(f"Expected scalar or [1] loss, got shape {tuple(loss.shape)}")
    return float(loss.item())


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
    total_rows: int,
    skipped_rows: int,
    shape_mismatch_rows: int,
    error_rows: int,
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

    return {
        "loss_name": loss_name,
        "target_name": "performance_degradation_abs_rel",
        "pearson": correlations[f"{prefix}_pearson"],
        "spearman": correlations[f"{prefix}_spearman"],
        "num_valid_pairs": int(valid_mask.sum().item()),
        "num_total_rows": int(total_rows),
        "num_skipped_rows": int(skipped_rows),
        "num_shape_mismatch_rows": int(shape_mismatch_rows),
        "num_error_rows": int(error_rows),
        "loss_mean": loss_mean,
        "loss_std": loss_std,
        "loss_min": loss_min,
        "loss_max": loss_max,
        "abs_rel_degradation_mean": deg_mean,
        "abs_rel_degradation_std": deg_std,
        "abs_rel_degradation_min": deg_min,
        "abs_rel_degradation_max": deg_max,
    }


@torch.inference_mode()
def collect_loss_values(
    dataset: FoundationCameraGroupedDataset,
    *,
    device: torch.device,
    max_rows: int | None,
    strict: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    log_losses: list[float] = []
    scale_shift_losses: list[float] = []
    abs_rel_degradations: list[float] = []
    counters = {
        "total_rows": 0,
        "skipped_rows": 0,
        "shape_mismatch_rows": 0,
        "error_rows": 0,
    }

    table = dataset.table
    if max_rows is not None:
        table = table.head(max_rows)

    progress = tqdm(
        table.iterrows(),
        total=len(table),
        desc="Computing losses",
        dynamic_ncols=True,
    )

    for _, row in progress:
        counters["total_rows"] += 1
        try:
            candidate_depth = dataset._load_depth(
                dataset._remap_path(row["source_depth_path"]),
                depth_scale=dataset.depth_scale,
            )
            canonical_depth = dataset._load_depth(
                dataset._remap_path(row["matched_depth_path"]),
                depth_scale=dataset.depth_scale,
            )

            if candidate_depth.shape != canonical_depth.shape:
                counters["shape_mismatch_rows"] += 1
                raise ValueError(
                    "Depth shape mismatch: "
                    f"candidate={tuple(candidate_depth.shape)}, "
                    f"canonical={tuple(canonical_depth.shape)}"
                )

            candidate_depth = candidate_depth.to(device=device)
            canonical_depth = canonical_depth.to(device=device)
            valid_mask = make_valid_mask(
                candidate_depth,
                canonical_depth,
                dataset.min_depth,
                dataset.max_depth,
            )

            candidate_depth = candidate_depth.unsqueeze(0)
            canonical_depth = canonical_depth.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)

            log_loss = log_scale_invariant_depth_difference(
                candidate_depth,
                canonical_depth,
                valid_mask,
            )
            scale_shift_loss = scale_shift_invariant_depth_loss(
                candidate_depth,
                canonical_depth,
                valid_mask,
            )

            log_losses.append(scalar_from_loss(log_loss))
            scale_shift_losses.append(scalar_from_loss(scale_shift_loss))
            abs_rel_degradations.append(float(row["performance_degradation_abs_rel"]))
        except Exception:
            counters["skipped_rows"] += 1
            counters["error_rows"] += 1
            if strict:
                raise

    if not log_losses:
        raise RuntimeError("No valid rows were evaluated. Check paths and filters.")

    return (
        torch.tensor(log_losses, dtype=torch.float32),
        torch.tensor(scale_shift_losses, dtype=torch.float32),
        torch.tensor(abs_rel_degradations, dtype=torch.float32),
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
    dataset = build_dataset(args, cfg)
    device = torch.device(args.device)

    log_losses, scale_shift_losses, degradations, counters = collect_loss_values(
        dataset,
        device=device,
        max_rows=args.max_rows,
        strict=args.strict,
    )

    rows = [
        summarize_loss_correlation(
            loss_name="log_scale_invariant_depth_difference",
            loss_values=log_losses,
            degradation_values=degradations,
            max_samples=args.correlation_max_samples,
            **counters,
        ),
        summarize_loss_correlation(
            loss_name="scale_shift_invariant_depth_loss",
            loss_values=scale_shift_losses,
            degradation_values=degradations,
            max_samples=args.correlation_max_samples,
            **counters,
        ),
    ]

    write_summary_csv(rows, args.output_csv)

    print(f"Saved correlation summary to {args.output_csv}")
    for row in rows:
        print(
            f"{row['loss_name']}: "
            f"pearson={format_float(row['pearson'])}, "
            f"spearman={format_float(row['spearman'])}, "
            f"n={row['num_valid_pairs']}"
        )


if __name__ == "__main__":
    main()
