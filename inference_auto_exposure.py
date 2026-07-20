from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from dataset.ae_dataset import (
    DEFAULT_AE_SCENE_PREFIX,
    AutoExposureMotionDataset,
    natural_key,
)
from evaluation_utils.eval_metrics import compute_comprehensive_depth_metrics
from evaluation_utils.eval_utils import align_relative_prediction_to_depth_space
from model.dav2_model import MODEL_IDS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/datasets/ATI/MDE/orbbec_realworld_dataset")
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "comparison_ae" / "ae_orbbec.csv"
MODEL_NAME = "small"
MODEL_ID = MODEL_IDS[MODEL_NAME]


@dataclass
class LapAccumulator:
    scene_name: str
    lap_id: str
    light: str
    collection_speed: str
    topology: str
    abs_rel_sum: float = 0.0
    a1_sum: float = 0.0
    evaluated_frames: int = 0
    skipped_frames: int = 0

    def update(self, abs_rel: float, a1: float) -> None:
        if not (math.isfinite(abs_rel) and math.isfinite(a1)):
            self.skipped_frames += 1
            return
        self.abs_rel_sum += abs_rel
        self.a1_sum += a1
        self.evaluated_frames += 1

    def skip(self) -> None:
        self.skipped_frames += 1

    def to_row(self) -> dict[str, object]:
        if self.evaluated_frames:
            mean_abs_rel = self.abs_rel_sum / self.evaluated_frames
            mean_a1 = self.a1_sum / self.evaluated_frames
        else:
            mean_abs_rel = float("nan")
            mean_a1 = float("nan")

        return {
            "scene_name": self.scene_name,
            "lap_id": self.lap_id,
            "light": self.light,
            "collection_speed": self.collection_speed,
            "topology": self.topology,
            "model_name": MODEL_NAME,
            "model_id": MODEL_ID,
            "alignment": "scale_shift",
            "evaluated_frames": self.evaluated_frames,
            "skipped_frames": self.skipped_frames,
            "abs_rel": mean_abs_rel,
            "a1": mean_a1,
        }


class IndexedDataset(Dataset):
    """Return the source index together with an existing dataset sample."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return index, self.dataset[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate relative Depth Anything V2 Small on AutoExposureMotionDataset "
            "with per-frame scale-shift alignment."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing comlab_scene_ae_* scene directories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path (default: comparison_ae/ae_orbbec.csv).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=518)
    parser.add_argument("--image-width", type=int, default=518)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument(
        "--min-valid-depth-ratio",
        type=float,
        default=0.0,
        help="Skip frames whose valid GT-depth ratio is below this value.",
    )
    parser.add_argument(
        "--scene-prefix",
        type=str,
        default=DEFAULT_AE_SCENE_PREFIX,
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.image_height <= 0 or args.image_width <= 0:
        parser.error("--image-height and --image-width must be positive")
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        parser.error("depth bounds must satisfy 0 < min-depth < max-depth")
    if not 0.0 <= args.min_valid_depth_ratio <= 1.0:
        parser.error("--min-valid-depth-ratio must be in [0, 1]")
    return args


def build_dataset_and_loader(
    args: argparse.Namespace,
    image_processor,
) -> tuple[AutoExposureMotionDataset, DataLoader]:
    dataset = AutoExposureMotionDataset(
        root_dir=str(args.dataset_root.expanduser().resolve()),
        image_processor=image_processor,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        min_length=1,
        scene_prefix=args.scene_prefix,
    )
    if not dataset.sequences:
        raise FileNotFoundError(
            "No auto-exposure laps were found under "
            f"{args.dataset_root}. Expected {args.scene_prefix}_*_*_topology* scenes."
        )

    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def get_accumulator(
    accumulators: dict[tuple[str, str], LapAccumulator],
    item,
) -> LapAccumulator:
    key = (item.scene_name, item.lap_id)
    if key not in accumulators:
        accumulators[key] = LapAccumulator(
            scene_name=item.scene_name,
            lap_id=item.lap_id,
            light=item.light,
            collection_speed=item.collection_speed,
            topology=item.topology,
        )
    return accumulators[key]


@torch.inference_mode()
def evaluate(
    model,
    dataset: AutoExposureMotionDataset,
    loader: DataLoader,
    device: torch.device,
    min_depth: float,
    max_depth: float,
    min_valid_depth_ratio: float,
    use_amp: bool,
) -> list[dict[str, object]]:
    accumulators: dict[tuple[str, str], LapAccumulator] = {}

    progress = tqdm(loader, desc="DAV2-small inference", dynamic_ncols=True)
    for item_indices, sample in progress:
        pixel_values, target_depth, valid_mask, _, _ = sample
        pixel_values = pixel_values.to(device=device, non_blocking=True)
        target_depth = target_depth.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        valid_mask = valid_mask.to(device=device, dtype=torch.bool, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred_relative = model(pixel_values=pixel_values).predicted_depth

        if pred_relative.ndim == 3:
            pred_relative = pred_relative.unsqueeze(1)
        elif pred_relative.ndim != 4:
            raise ValueError(
                "Expected predicted_depth shaped [B,H,W] or [B,1,H,W], "
                f"got {tuple(pred_relative.shape)}"
            )

        pred_relative = F.interpolate(
            pred_relative.float(),
            size=target_depth.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        target_depth = target_depth.unsqueeze(1)
        valid_mask = valid_mask.unsqueeze(1)

        # DAV2 relative output is inverse-depth-like. Fit scale and shift to
        # inverse GT independently for every frame, then evaluate in depth space.
        aligned_depth = align_relative_prediction_to_depth_space(
            pred=pred_relative,
            gt=target_depth,
            valid_mask=valid_mask,
            align_mode="scale_shift",
        )["depth"]
        metrics = compute_comprehensive_depth_metrics(
            mu=aligned_depth,
            target=target_depth,
            valid_mask=valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        abs_rel_values = metrics["abs_rel"].detach().cpu()
        a1_values = metrics["a1"].detach().cpu()
        valid_ratios = valid_mask.float().flatten(1).mean(dim=1).detach().cpu()

        for offset, dataset_index in enumerate(item_indices.tolist()):
            item = dataset.items[dataset_index]
            accumulator = get_accumulator(accumulators, item)
            valid_ratio = float(valid_ratios[offset].item())
            if valid_ratio == 0.0 or valid_ratio < min_valid_depth_ratio:
                accumulator.skip()
                continue
            accumulator.update(
                abs_rel=float(abs_rel_values[offset].item()),
                a1=float(a1_values[offset].item()),
            )

        if item_indices.numel():
            last_item = dataset.items[int(item_indices[-1].item())]
            progress.set_postfix(scene=last_item.scene_name, lap=last_item.lap_id)

    return [
        accumulator.to_row()
        for _, accumulator in sorted(
            accumulators.items(),
            key=lambda pair: (
                natural_key(pair[1].scene_name),
                natural_key(pair[1].lap_id),
            ),
        )
    ]


def write_csv(rows: Sequence[dict[str, object]], output_csv: Path) -> Path:
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_name",
        "lap_id",
        "light",
        "collection_speed",
        "topology",
        "model_name",
        "model_id",
        "alignment",
        "evaluated_frames",
        "skipped_frames",
        "abs_rel",
        "a1",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def print_results(rows: Sequence[dict[str, object]]) -> None:
    print("\nPer scene/lap mean metrics (scale-shift aligned)")
    print(f"{'scene':<48} {'lap':<32} {'frames':>8} {'AbsRel':>10} {'A1':>10}")
    print("-" * 114)
    for row in rows:
        print(
            f"{str(row['scene_name']):<48} "
            f"{str(row['lap_id']):<32} "
            f"{int(row['evaluated_frames']):>8d} "
            f"{float(row['abs_rel']):>10.4f} "
            f"{float(row['a1']):>10.4f}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    cache_dir = (
        str(args.hf_cache_dir.expanduser().resolve())
        if args.hf_cache_dir is not None
        else None
    )
    print(f"Loading {MODEL_ID} on {device} ...")
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
    ).to(device)
    model.eval()

    dataset, loader = build_dataset_and_loader(args, image_processor)
    print(
        f"Found {len(dataset):,} frames in {len(dataset.sequences):,} laps; "
        "running per-frame scale-shift aligned evaluation."
    )
    rows = evaluate(
        model=model,
        dataset=dataset,
        loader=loader,
        device=device,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        use_amp=device.type == "cuda" and not args.no_amp,
    )
    print_results(rows)
    output_csv = write_csv(rows, args.output_csv)
    print(f"\nSaved {len(rows)} scene/lap rows to {output_csv}")


if __name__ == "__main__":
    main()
