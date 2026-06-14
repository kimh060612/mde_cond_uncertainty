import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import torch
from transformers import AutoImageProcessor

from ati_dataset import ATIRealWorldDepthDataset
from dav2_ati_model import ConditionedGaussianDepthAnythingV2, MODEL_IDS
from eval_utils import (
    compute_metrics,
    compute_relative_depth_metrics,
    _accumulate_finite_metrics,
    _mean_finite_metrics,
)
from loss_fn import image_uncertainty_score


def _natural_key(value):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(value))
    ]


def _format_condition_key(exposure, gain):
    return f"exposure_{exposure:g}_gain_{gain:g}"


def _finite_mean(values, default=float("nan")):
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return default
    return float(np.mean(finite_values))


def _merge_metric_accumulators(dst_sums, dst_counts, src_sums, src_counts):
    for key, value in src_sums.items():
        dst_sums[key] = dst_sums.get(key, 0.0) + value
        dst_counts[key] = dst_counts.get(key, 0) + src_counts.get(key, 0)


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
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


def _processor_source(args):
    if args.processor_dir is not None:
        return args.processor_dir

    checkpoint_dir = Path(args.ckpt).resolve().parent
    if (checkpoint_dir / "preprocessor_config.json").is_file():
        return str(checkpoint_dir)

    return MODEL_IDS[args.model]


def _metadata_list(metadata, key, fallback):
    value = metadata.get(key)
    if value:
        return list(value)
    return list(fallback)


def _metadata_float(metadata, key, fallback):
    value = metadata.get(key)
    if value is None:
        return fallback
    return float(value)


def _build_sequence_index(dataset):
    scenes = {}
    grouped = defaultdict(lambda: defaultdict(dict))

    for idx, item in enumerate(dataset.items):
        scenes[item.scene_name] = {
            "scene_name": item.scene_name,
            "scene_prefix": item.scene_prefix,
            "light": item.light,
            "speed": item.speed,
        }
        grouped[item.scene_name][(item.exposure, item.gain)][item.frame_id] = idx

    return scenes, grouped


def _stack_samples(dataset, indices):
    samples = [dataset[index] for index in indices]
    return (
        torch.stack([sample[0] for sample in samples], dim=0),
        torch.stack([sample[1] for sample in samples], dim=0),
        torch.stack([sample[2] for sample in samples], dim=0),
        torch.stack([sample[3] for sample in samples], dim=0),
        torch.stack([sample[4] for sample in samples], dim=0),
    )


def _select_uncertainty_tensor(output, uncertainty_kind):
    if uncertainty_kind == "std":
        return output["std"]
    if uncertainty_kind == "var":
        return output["var"]
    if uncertainty_kind == "log_var":
        return output["log_var"]
    raise ValueError(f"Unsupported uncertainty kind: {uncertainty_kind}")


@torch.no_grad()
def _evaluate_batch(
    model_id,
    model,
    batch,
    device,
    amp,
    uncertainty_mode,
    uncertainty_kind,
    selection_mask,
    min_depth,
    max_depth,
    relative_align_mode,
):
    pixel_values, depth, valid_mask, condition, _condition_stats = batch
    pixel_values = pixel_values.to(device, non_blocking=True)
    depth = depth.to(device, non_blocking=True)
    valid_mask = valid_mask.to(device, non_blocking=True)
    condition = condition.to(device, non_blocking=True)

    target_size = depth.shape[-2:]
    with torch.autocast(device_type=device.type, enabled=amp):
        output = model(
            pixel_values,
            condition=condition,
            target_size=target_size,
        )

    uncertainty = _select_uncertainty_tensor(output, uncertainty_kind).detach().float()
    if selection_mask == "valid_depth":
        uncertainty_mask = valid_mask
    else:
        uncertainty_mask = torch.ones_like(valid_mask)

    image_scores = image_uncertainty_score(
        uncertainty,
        uncertainty_mask,
        mode=uncertainty_mode,
    )

    metric_sums = {}
    metric_counts = {}
    for batch_idx in range(depth.shape[0]):
        if model_id.startswith("metric"):
            image_metrics = compute_metrics(
                output["mu"][batch_idx : batch_idx + 1].detach().float(),
                depth[batch_idx : batch_idx + 1],
                valid_mask[batch_idx : batch_idx + 1],
            )
        else:
            image_metrics = compute_relative_depth_metrics(
                output["mu"][batch_idx : batch_idx + 1].detach().float(),
                depth[batch_idx : batch_idx + 1],
                valid_mask[batch_idx : batch_idx + 1],
                min_depth=min_depth,
                max_depth=max_depth,
                align_mode=relative_align_mode,
            )
        _accumulate_finite_metrics(metric_sums, metric_counts, image_metrics)

    return image_scores.detach().cpu().tolist(), metric_sums, metric_counts


def _evaluate_candidate_window(
    dataset,
    indices,
    model_id,
    model,
    device,
    amp,
    eval_batch_size,
    uncertainty_mode,
    uncertainty_kind,
    selection_mask,
    min_depth,
    max_depth,
    relative_align_mode,
):
    metric_sums = {}
    metric_counts = {}
    uncertainty_scores = []

    for offset in range(0, len(indices), eval_batch_size):
        batch_indices = indices[offset : offset + eval_batch_size]
        batch = _stack_samples(dataset, batch_indices)
        batch_scores, batch_sums, batch_counts = _evaluate_batch(
            model_id=model_id,
            model=model,
            batch=batch,
            device=device,
            amp=amp,
            uncertainty_mode=uncertainty_mode,
            uncertainty_kind=uncertainty_kind,
            selection_mask=selection_mask,
            min_depth=min_depth,
            max_depth=max_depth,
            relative_align_mode=relative_align_mode,
        )
        uncertainty_scores.extend(batch_scores)
        _merge_metric_accumulators(metric_sums, metric_counts, batch_sums, batch_counts)

    metrics = _mean_finite_metrics(metric_sums, metric_counts)
    return {
        "uncertainty_score": _finite_mean(uncertainty_scores, default=float("inf")),
        "uncertainty_scores": uncertainty_scores,
        "metrics": metrics,
        "metric_sums": metric_sums,
        "metric_counts": metric_counts,
        "num_frames": len(indices),
    }


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _row_metric_keys(rows):
    metric_keys = set()
    for row in rows:
        metric_keys.update(key for key in row if key.startswith("metric_"))
    return sorted(metric_keys)


def _evaluate_sequences(args, dataset, model_id, model, device, amp):
    scene_info, grouped = _build_sequence_index(dataset)
    window_rows = []
    candidate_rows = []
    scene_accumulators = {}
    overall_sums = {}
    overall_counts = {}

    for scene_name in sorted(grouped, key=_natural_key):
        exposure_groups = grouped[scene_name]
        if not exposure_groups:
            continue

        common_frame_ids = None
        for frame_map in exposure_groups.values():
            frame_ids = set(frame_map)
            common_frame_ids = frame_ids if common_frame_ids is None else common_frame_ids & frame_ids

        if not common_frame_ids:
            print(f"[skip] {scene_name}: no common frame ids across exposure/gain candidates")
            continue

        frame_ids = sorted(common_frame_ids, key=_natural_key)
        windows = [
            frame_ids[offset : offset + args.window_size]
            for offset in range(0, len(frame_ids), args.window_size)
        ]
        if args.drop_last:
            windows = [window for window in windows if len(window) == args.window_size]
        if args.max_windows_per_scene is not None:
            windows = windows[: args.max_windows_per_scene]

        scene_meta = scene_info[scene_name]
        scene_accumulators[scene_name] = {
            "info": scene_meta,
            "metric_sums": {},
            "metric_counts": {},
            "windows": 0,
            "frames": 0,
            "selection_counts": defaultdict(int),
            "selected_uncertainties": [],
        }

        print(
            f"[scene] {scene_name}: {len(exposure_groups)} candidates, "
            f"{len(frame_ids)} common frames, {len(windows)} windows"
        )

        for window_idx, window_frame_ids in enumerate(windows):
            if not window_frame_ids:
                continue

            candidate_results = []
            for exposure, gain in sorted(exposure_groups, key=lambda key: (key[0], key[1])):
                frame_map = exposure_groups[(exposure, gain)]
                indices = [frame_map[frame_id] for frame_id in window_frame_ids]
                result = _evaluate_candidate_window(
                    dataset=dataset,
                    indices=indices,
                    model_id=model_id,
                    model=model,
                    device=device,
                    amp=amp,
                    eval_batch_size=args.eval_batch_size,
                    uncertainty_mode=args.uncertainty_mode,
                    uncertainty_kind=args.uncertainty_kind,
                    selection_mask=args.selection_mask,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    relative_align_mode=args.relative_align_mode,
                )
                candidate_results.append(
                    {
                        "exposure": exposure,
                        "gain": gain,
                        **result,
                    }
                )

            finite_candidates = [
                candidate
                for candidate in candidate_results
                if math.isfinite(candidate["uncertainty_score"])
            ]
            if not finite_candidates:
                print(f"[skip] {scene_name} window {window_idx}: no finite uncertainty scores")
                continue

            selected = min(
                finite_candidates,
                key=lambda candidate: (
                    candidate["uncertainty_score"],
                    candidate["exposure"],
                    candidate["gain"],
                ),
            )
            selected_key = _format_condition_key(selected["exposure"], selected["gain"])

            acc = scene_accumulators[scene_name]
            acc["windows"] += 1
            acc["frames"] += selected["num_frames"]
            acc["selection_counts"][selected_key] += 1
            acc["selected_uncertainties"].append(selected["uncertainty_score"])
            _merge_metric_accumulators(
                acc["metric_sums"],
                acc["metric_counts"],
                selected["metric_sums"],
                selected["metric_counts"],
            )
            _merge_metric_accumulators(
                overall_sums,
                overall_counts,
                selected["metric_sums"],
                selected["metric_counts"],
            )

            base_row = {
                **scene_meta,
                "window_index": window_idx,
                "frame_start": window_frame_ids[0],
                "frame_end": window_frame_ids[-1],
                "num_frames": selected["num_frames"],
                "selected_exposure": selected["exposure"],
                "selected_gain": selected["gain"],
                "selected_condition": selected_key,
                "selected_uncertainty": selected["uncertainty_score"],
                "uncertainty_kind": args.uncertainty_kind,
                "uncertainty_mode": args.uncertainty_mode,
                "selection_mask": args.selection_mask,
            }
            for key, value in selected["metrics"].items():
                base_row[f"metric_{key}"] = value
            window_rows.append(base_row)

            for candidate in candidate_results:
                candidate_key = _format_condition_key(candidate["exposure"], candidate["gain"])
                candidate_row = {
                    **scene_meta,
                    "window_index": window_idx,
                    "frame_start": window_frame_ids[0],
                    "frame_end": window_frame_ids[-1],
                    "num_frames": candidate["num_frames"],
                    "exposure": candidate["exposure"],
                    "gain": candidate["gain"],
                    "condition": candidate_key,
                    "uncertainty_score": candidate["uncertainty_score"],
                    "selected": int(candidate is selected),
                }
                for key, value in candidate["metrics"].items():
                    candidate_row[f"metric_{key}"] = value
                candidate_rows.append(candidate_row)

            print(
                f"  window {window_idx:04d}: selected {selected_key} "
                f"unc={selected['uncertainty_score']:.6f} "
                f"metrics={selected['metrics']}"
            )

    scene_summary_rows = []
    for scene_name in sorted(scene_accumulators, key=_natural_key):
        acc = scene_accumulators[scene_name]
        summary = {
            **acc["info"],
            "windows": acc["windows"],
            "frames": acc["frames"],
            "mean_selected_uncertainty": _finite_mean(acc["selected_uncertainties"]),
            "selection_counts": json.dumps(dict(sorted(acc["selection_counts"].items()))),
        }
        for key, value in _mean_finite_metrics(acc["metric_sums"], acc["metric_counts"]).items():
            summary[f"metric_{key}"] = value
        scene_summary_rows.append(summary)

    overall_summary = {
        "num_scenes": len([row for row in scene_summary_rows if row["windows"] > 0]),
        "num_windows": sum(row["windows"] for row in scene_summary_rows),
        "num_frames": sum(row["frames"] for row in scene_summary_rows),
        "metrics": _mean_finite_metrics(overall_sums, overall_counts),
    }

    return window_rows, candidate_rows, scene_summary_rows, overall_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/media/michael/ssd1/AIoT_ATI/realworld_dataset")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="metric-indoor-small", choices=list(MODEL_IDS.keys()))
    parser.add_argument("--processor_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./sequence_eval_results")

    parser.add_argument("--image_height", type=int, default=518)
    parser.add_argument("--image_width", type=int, default=518)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--max_windows_per_scene", type=int, default=None)

    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_valid_depth_ratio", type=float, default=0.0)
    parser.add_argument("--min_log_var", type=float, default=-5.0)
    parser.add_argument("--max_log_var", type=float, default=3.0)
    parser.add_argument("--uncertainty_width", type=int, default=64)
    parser.add_argument("--uncertainty_blocks", type=int, default=3)
    parser.add_argument("--uncertainty_dropout", type=float, default=0.05)
    parser.add_argument("--uncertainty_kind", type=str, default="std", choices=["std", "var", "log_var"])
    parser.add_argument("--uncertainty_mode", type=str, default="top20", choices=["mean", "top10", "top20"])
    parser.add_argument("--selection_mask", type=str, default="all", choices=["all", "valid_depth"])

    parser.add_argument("--light_levels", nargs="*", default=["dark", "dim", "normal"])
    parser.add_argument("--speed_levels", nargs="*", default=["slow", "fast"])
    parser.add_argument("--scene_prefixes", nargs="*", default=["comlab_scene2", "realsense_scene"])
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--relative_align_mode", type=str, default="scale_shift", choices=["median", "scale_shift"])
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--non_strict_load", action="store_true")

    return parser.parse_args()


def main(args):
    if args.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {args.window_size}")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {args.eval_batch_size}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, state_dict, metadata = _load_checkpoint(args.ckpt)
    model_id = MODEL_IDS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") and (not args.no_amp)

    light_levels = _metadata_list(metadata, "light_levels", args.light_levels)
    speed_levels = _metadata_list(metadata, "speed_levels", args.speed_levels)

    image_processor = AutoImageProcessor.from_pretrained(
        _processor_source(args),
        cache_dir=args.hf_cache_dir,
    )

    dataset = ATIRealWorldDepthDataset(
        root_dir=args.dataset_root,
        image_processor=image_processor,
        image_size=(args.image_height, args.image_width),
        split="all",
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        light_levels=light_levels,
        speed_levels=speed_levels,
        scene_prefixes=args.scene_prefixes,
    )

    dataset.exposure_min = _metadata_float(metadata, "exposure_min", dataset.exposure_min)
    dataset.exposure_max = _metadata_float(metadata, "exposure_max", dataset.exposure_max)
    dataset.gain_min = _metadata_float(metadata, "gain_min", dataset.gain_min)
    dataset.gain_max = _metadata_float(metadata, "gain_max", dataset.gain_max)

    cond_dim = len(metadata.get("condition_names", dataset.condition_names))
    if cond_dim != dataset.condition_dim:
        raise ValueError(
            f"Checkpoint condition dim ({cond_dim}) does not match dataset condition dim "
            f"({dataset.condition_dim}). Check light/speed level metadata."
        )

    model = ConditionedGaussianDepthAnythingV2(
        model_id=model_id,
        cond_dim=cond_dim,
        cache_dir=args.hf_cache_dir,
        freeze_backbone=False,
        min_log_var=args.min_log_var,
        max_log_var=args.max_log_var,
        uncertainty_width=args.uncertainty_width,
        uncertainty_blocks=args.uncertainty_blocks,
        uncertainty_dropout=args.uncertainty_dropout,
    ).to(device)

    load_result = model.load_state_dict(state_dict, strict=not args.non_strict_load)
    model.eval()

    print(f"Using model: {model_id}")
    print(f"checkpoint: {args.ckpt}")
    print(f"dataset root: {args.dataset_root}")
    print(f"samples: {len(dataset):,}")
    print(f"condition names: {list(dataset.condition_names)}")
    if args.non_strict_load:
        print(f"load_state_dict result: {load_result}")

    window_rows, candidate_rows, scene_summary_rows, overall_summary = _evaluate_sequences(
        args=args,
        dataset=dataset,
        model_id=args.model,
        model=model,
        device=device,
        amp=amp,
    )

    window_fields = [
        "scene_name",
        "scene_prefix",
        "light",
        "speed",
        "window_index",
        "frame_start",
        "frame_end",
        "num_frames",
        "selected_exposure",
        "selected_gain",
        "selected_condition",
        "selected_uncertainty",
        "uncertainty_kind",
        "uncertainty_mode",
        "selection_mask",
        *_row_metric_keys(window_rows),
    ]
    candidate_fields = [
        "scene_name",
        "scene_prefix",
        "light",
        "speed",
        "window_index",
        "frame_start",
        "frame_end",
        "num_frames",
        "exposure",
        "gain",
        "condition",
        "uncertainty_score",
        "selected",
        *_row_metric_keys(candidate_rows),
    ]
    scene_fields = [
        "scene_name",
        "scene_prefix",
        "light",
        "speed",
        "windows",
        "frames",
        "mean_selected_uncertainty",
        "selection_counts",
        *_row_metric_keys(scene_summary_rows),
    ]

    _write_csv(output_dir / "sequence_window_results.csv", window_rows, window_fields)
    _write_csv(output_dir / "sequence_candidate_results.csv", candidate_rows, candidate_fields)
    _write_csv(output_dir / "scene_summary.csv", scene_summary_rows, scene_fields)

    summary_payload = {
        "args": vars(args),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_val_metrics": checkpoint.get("val_metrics") if isinstance(checkpoint, dict) else None,
        "dataset_metadata": metadata,
        "overall": overall_summary,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"saved window results: {output_dir / 'sequence_window_results.csv'}")
    print(f"saved candidate results: {output_dir / 'sequence_candidate_results.csv'}")
    print(f"saved scene summary: {output_dir / 'scene_summary.csv'}")
    print(f"overall: {overall_summary}")


if __name__ == "__main__":
    main(parse_args())
