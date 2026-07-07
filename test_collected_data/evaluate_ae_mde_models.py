from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.ae_dataset import (  # noqa: E402
    DEFAULT_AE_SCENE_PREFIX,
    AutoExposureMotionDataset,
    natural_key,
)
from dataset.ati_dataset_refactored import LIGHT_LEVELS, MOTION_LEVELS  # noqa: E402
from evaluation_utils.eval_metrics import compute_comprehensive_depth_metrics  # noqa: E402
from evaluation_utils.eval_utils import (  # noqa: E402
    align_relative_prediction_to_depth_space,
    ensure_bchw,
)


DEFAULT_MODEL_IDS = (
    "depth-anything/da3metric-large",
    "depth-anything/da3-base",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    depth_type: str


@dataclass
class LapAccumulator:
    scene_name: str
    lap_id: str
    model_name: str
    model_id: str
    depth_type: str
    light: str
    collection_speed: str
    topology: str
    abs_rel_sum: float = 0.0
    abs_rel_count: int = 0
    delta_1_25_sum: float = 0.0
    delta_1_25_count: int = 0
    valid_pixels: int = 0
    total_pixels: int = 0
    frames: int = 0
    skipped_frames: int = 0

    def update(
        self,
        abs_rel: float,
        delta_1_25: float,
        valid_pixels: int,
        total_pixels: int,
    ) -> None:
        if math.isfinite(abs_rel):
            self.abs_rel_sum += abs_rel
            self.abs_rel_count += 1
        if math.isfinite(delta_1_25):
            self.delta_1_25_sum += delta_1_25
            self.delta_1_25_count += 1
        self.valid_pixels += int(valid_pixels)
        self.total_pixels += int(total_pixels)
        self.frames += 1

    def mark_skipped(self) -> None:
        self.skipped_frames += 1

    def to_row(self) -> dict[str, object]:
        abs_rel = (
            self.abs_rel_sum / self.abs_rel_count
            if self.abs_rel_count > 0
            else float("nan")
        )
        delta_1_25 = (
            self.delta_1_25_sum / self.delta_1_25_count
            if self.delta_1_25_count > 0
            else float("nan")
        )
        valid_ratio = (
            self.valid_pixels / self.total_pixels
            if self.total_pixels > 0
            else float("nan")
        )
        return {
            "scene_name": self.scene_name,
            "lap_id": self.lap_id,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "depth_type": self.depth_type,
            "light": self.light,
            "collection_speed": self.collection_speed,
            "topology": self.topology,
            "frames": self.frames,
            "skipped_frames": self.skipped_frames,
            "valid_pixels": self.valid_pixels,
            "total_pixels": self.total_pixels,
            "valid_pixel_ratio": valid_ratio,
            "abs_rel": abs_rel,
            "delta_1_25": delta_1_25,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Depth Anything 3 models on AE-captured "
            "comlab_scene_ae_*_*_topology* laps."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing comlab_scene_ae_*_*_topology* scenes.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("evaluation_models") / "ae_da3_mde_metrics.csv",
        help="CSV path for per scene/lap/model metrics.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_IDS),
        help=(
            "Depth Anything 3 model ids. Models whose id/name contains "
            "'metric' are evaluated as metric depth; others are relative."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of frames passed to DA3 inference at once.",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="DA3 input processor target resolution.",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=(
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        ),
        help="DA3 input processor resize/crop policy.",
    )
    parser.add_argument(
        "--relative-align-mode",
        type=str,
        default="scale_shift",
        choices=("scale_shift", "median"),
        help="Alignment mode for relative-depth models.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=1e-3,
        help="Minimum valid/evaluated depth in meters.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum valid/evaluated depth in meters.",
    )
    parser.add_argument(
        "--min-valid-depth-ratio",
        type=float,
        default=0.0,
        help="Skip frames below this valid-depth ratio.",
    )
    parser.add_argument(
        "--scene-prefix",
        type=str,
        default=DEFAULT_AE_SCENE_PREFIX,
        help="AE scene prefix to scan.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for DA3 inference and metric computation.",
    )
    parser.add_argument(
        "--max-laps",
        type=int,
        default=None,
        help="Optional debug limit for the number of laps evaluated.",
    )
    parser.add_argument(
        "--max-frames-per-lap",
        type=int,
        default=None,
        help="Optional debug limit for frames per lap.",
    )
    return parser.parse_args()


def load_da3_model(model_id: str, device: torch.device):
    try:
        from depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "depth_anything_3 is required for this evaluator. "
            "Install Depth Anything 3 in the environment used to run this script."
        ) from exc

    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    model.eval()
    return model


def make_model_specs(model_ids: Sequence[str]) -> List[ModelSpec]:
    specs = []
    for model_id in model_ids:
        normalized = model_id.rstrip("/")
        name = normalized.split("/")[-1]
        depth_type = "metric" if "metric" in normalized.lower() else "relative"
        specs.append(ModelSpec(name=name, model_id=model_id, depth_type=depth_type))
    return specs


def batched(values: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def resize_depth_nearest(depth: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    depth_tensor = torch.from_numpy(depth.astype(np.float32, copy=False))
    if depth_tensor.ndim == 3:
        depth_tensor = depth_tensor.squeeze()
    depth_tensor = ensure_bchw(depth_tensor.unsqueeze(0))
    resized = F.interpolate(depth_tensor, size=size, mode="nearest")
    return resized.squeeze(0).squeeze(0)


def load_target_batch(
    dataset: AutoExposureMotionDataset,
    item_indices: Sequence[int],
    pred_hw: tuple[int, int],
    min_depth: float,
    max_depth: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth_tensors = []
    valid_masks = []
    valid_ratios = []

    for item_index in item_indices:
        item = dataset.items[item_index]
        depth = dataset._load_depth(item.depth_path)
        depth_tensor = resize_depth_nearest(depth, pred_hw)
        valid_mask = torch.isfinite(depth_tensor)
        valid_mask &= depth_tensor > min_depth
        valid_mask &= depth_tensor < max_depth
        depth_tensor = torch.nan_to_num(
            depth_tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        depth_tensors.append(depth_tensor)
        valid_masks.append(valid_mask)
        valid_ratios.append(valid_mask.float().mean())

    depth_batch = torch.stack(depth_tensors, dim=0).to(device=device, dtype=torch.float32)
    mask_batch = torch.stack(valid_masks, dim=0).to(device=device, dtype=torch.bool)
    ratio_batch = torch.stack(valid_ratios, dim=0).to(device=device, dtype=torch.float32)
    return depth_batch, mask_batch, ratio_batch


def prepare_prediction_depth(prediction) -> np.ndarray:
    depth = np.asarray(prediction.depth, dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[None, ...]
    if depth.ndim != 3:
        raise ValueError(f"Expected DA3 depth with shape [N, H, W], got {depth.shape}")
    return depth


def evaluate_depth_batch(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_type: str,
    relative_align_mode: str,
    min_depth: float,
    max_depth: float,
) -> dict[str, torch.Tensor]:
    pred_depth = ensure_bchw(pred_depth)
    target_depth = ensure_bchw(target_depth)
    valid_mask = ensure_bchw(valid_mask).bool()

    if depth_type == "relative":
        aligned = align_relative_prediction_to_depth_space(
            pred_depth,
            target_depth,
            valid_mask,
            align_mode=relative_align_mode,
        )
        eval_depth = aligned["depth"]
    elif depth_type == "metric":
        eval_depth = pred_depth
    else:
        raise ValueError(f"Unknown depth_type: {depth_type}")

    return compute_comprehensive_depth_metrics(
        mu=eval_depth,
        target=target_depth,
        valid_mask=valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def accumulator_key(item, spec: ModelSpec) -> tuple[str, str, str]:
    return item.scene_name, item.lap_id, spec.name


def get_accumulator(
    accumulators: dict[tuple[str, str, str], LapAccumulator],
    item,
    spec: ModelSpec,
) -> LapAccumulator:
    key = accumulator_key(item, spec)
    if key not in accumulators:
        accumulators[key] = LapAccumulator(
            scene_name=item.scene_name,
            lap_id=item.lap_id,
            model_name=spec.name,
            model_id=spec.model_id,
            depth_type=spec.depth_type,
            light=item.light,
            collection_speed=item.collection_speed,
            topology=item.topology,
        )
    return accumulators[key]


def update_accumulators(
    accumulators: dict[tuple[str, str, str], LapAccumulator],
    dataset: AutoExposureMotionDataset,
    spec: ModelSpec,
    item_indices: Sequence[int],
    metrics: dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    valid_ratio: torch.Tensor,
    min_valid_depth_ratio: float,
) -> None:
    abs_rel = metrics["abs_rel"].detach().cpu()
    delta_1_25 = metrics["a1"].detach().cpu()
    valid_mask_cpu = valid_mask.detach().cpu()
    valid_ratio_cpu = valid_ratio.detach().cpu()

    for batch_offset, item_index in enumerate(item_indices):
        item = dataset.items[item_index]
        accumulator = get_accumulator(accumulators, item, spec)
        if float(valid_ratio_cpu[batch_offset].item()) < min_valid_depth_ratio:
            accumulator.mark_skipped()
            continue
        frame_mask = valid_mask_cpu[batch_offset]
        accumulator.update(
            abs_rel=float(abs_rel[batch_offset].item()),
            delta_1_25=float(delta_1_25[batch_offset].item()),
            valid_pixels=int(frame_mask.sum().item()),
            total_pixels=int(frame_mask.numel()),
        )


def evaluate_model(
    model,
    spec: ModelSpec,
    dataset: AutoExposureMotionDataset,
    device: torch.device,
    batch_size: int,
    process_res: int,
    process_res_method: str,
    relative_align_mode: str,
    min_depth: float,
    max_depth: float,
    min_valid_depth_ratio: float,
    max_laps: int | None,
    max_frames_per_lap: int | None,
) -> list[dict[str, object]]:
    accumulators: dict[tuple[str, str, str], LapAccumulator] = {}
    sequences = sorted(
        dataset.sequences,
        key=lambda sequence: (
            natural_key(sequence.scene_name),
            natural_key(sequence.lap_id),
        ),
    )
    if max_laps is not None:
        sequences = sequences[: max(0, max_laps)]

    progress = tqdm(
        sequences,
        desc=f"Evaluating {spec.name}",
        dynamic_ncols=True,
    )
    for sequence in progress:
        if not sequence.dataset_indices:
            continue

        item_indices = list(sequence.dataset_indices)
        if max_frames_per_lap is not None:
            item_indices = item_indices[: max(0, max_frames_per_lap)]

        for batch_indices in batched(item_indices, batch_size):
            image_paths = [
                str(dataset.items[item_index].rgb_path)
                for item_index in batch_indices
            ]
            prediction = model.inference(
                image_paths,
                export_dir=None,
                export_format="mini_npz",
                process_res=process_res,
                process_res_method=process_res_method,
            )
            pred_np = prepare_prediction_depth(prediction)
            if pred_np.shape[0] != len(batch_indices):
                raise RuntimeError(
                    "DA3 returned a different number of depth maps than inputs: "
                    f"{pred_np.shape[0]} vs {len(batch_indices)}"
                )

            pred_depth = torch.from_numpy(pred_np).to(device=device, dtype=torch.float32)
            pred_hw = (int(pred_depth.shape[-2]), int(pred_depth.shape[-1]))
            target_depth, valid_mask, valid_ratio = load_target_batch(
                dataset=dataset,
                item_indices=batch_indices,
                pred_hw=pred_hw,
                min_depth=min_depth,
                max_depth=max_depth,
                device=device,
            )
            metrics = evaluate_depth_batch(
                pred_depth=pred_depth,
                target_depth=target_depth,
                valid_mask=valid_mask,
                depth_type=spec.depth_type,
                relative_align_mode=relative_align_mode,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            update_accumulators(
                accumulators=accumulators,
                dataset=dataset,
                spec=spec,
                item_indices=batch_indices,
                metrics=metrics,
                valid_mask=valid_mask,
                valid_ratio=valid_ratio,
                min_valid_depth_ratio=min_valid_depth_ratio,
            )

        progress.set_postfix(scene=sequence.scene_name, lap=sequence.lap_id)

    return [
        accumulator.to_row()
        for _, accumulator in sorted(
            accumulators.items(),
            key=lambda pair: (
                natural_key(pair[1].scene_name),
                natural_key(pair[1].lap_id),
                pair[1].model_name,
            ),
        )
    ]


def write_csv(rows: Sequence[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_name",
        "lap_id",
        "model_name",
        "model_id",
        "depth_type",
        "light",
        "collection_speed",
        "topology",
        "frames",
        "skipped_frames",
        "valid_pixels",
        "total_pixels",
        "valid_pixel_ratio",
        "abs_rel",
        "delta_1_25",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    specs = make_model_specs(args.models)

    dataset = AutoExposureMotionDataset(
        root_dir=str(args.dataset_root),
        image_processor=None,
        image_size=None,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        light_levels=LIGHT_LEVELS,
        speed_levels=MOTION_LEVELS,
        min_length=1,
        scene_prefix=args.scene_prefix,
    )
    if not dataset.sequences:
        raise FileNotFoundError(
            "No AE laps found. Expected directories matching "
            f"{args.scene_prefix}_*_*_topology* under {args.dataset_root}"
        )

    all_rows = []
    for spec in specs:
        model = load_da3_model(spec.model_id, device=device)
        rows = evaluate_model(
            model=model,
            spec=spec,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            relative_align_mode=args.relative_align_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            min_valid_depth_ratio=args.min_valid_depth_ratio,
            max_laps=args.max_laps,
            max_frames_per_lap=args.max_frames_per_lap,
        )
        all_rows.extend(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(all_rows, args.output_csv)
    print(f"Wrote {len(all_rows)} scene/lap/model rows to {args.output_csv}")


if __name__ == "__main__":
    main()
