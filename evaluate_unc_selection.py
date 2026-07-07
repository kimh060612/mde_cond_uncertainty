import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import AutoImageProcessor

from dataset.ae_dataset import (
    AutoExposureMotionDataset,
    MotionSequence,
    SceneKey,
    motion_features_from_metadata,
    natural_key,
    topology_id,
)
from dataset.ati_dataset_refactored import (
    ATIRealWorldUncertaintyDataset,
    ATIRealWorldUncertaintyValidationDataset,
    LIGHT_LEVELS,
    MOTION_LEVELS,
)
from evaluation_utils.eval_metrics import compute_comprehensive_depth_metrics
from evaluation_utils.eval_utils import align_relative_prediction_to_depth_space
from model.dav2_ati_bias_model import FrozenDepthCameraGaussian
from model.dav2_ati_model import MODEL_IDS
from model.loss_fn import image_uncertainty_score
from utils.train_utils import copy_condition_normalization, seed_everything


DEFAULT_DATASET_ROOT = "/issac-sim/dataset/realworld_dataset"


@dataclass
class AlignedWindow:
    sequence: MotionSequence
    start: int
    end: int
    cost: float

    @property
    def frame_ids(self):
        return self.sequence.frame_ids[self.start : self.end]


def _cfg_select(cfg: DictConfig, key: str, default=None):
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def _cfg_list(cfg: DictConfig, key: str, default: Sequence):
    value = OmegaConf.select(cfg, key)
    return list(default) if value is None else list(value)


def _format_condition_key(exposure: float, gain: float) -> str:
    return f"exposure_{float(exposure):g}_gain_{float(gain):g}"


def _finite_mean(values, default=float("nan")):
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return default
    return float(np.mean(finite_values))


def _merge_metric_accumulators(dst_sums, dst_counts, src_sums, src_counts):
    for key, value in src_sums.items():
        dst_sums[key] = dst_sums.get(key, 0.0) + value
        dst_counts[key] = dst_counts.get(key, 0) + src_counts.get(key, 0)


def _mean_metric_accumulator(metric_sums, metric_counts):
    return {
        key: metric_sums[key] / metric_counts[key]
        for key in sorted(metric_sums)
        if metric_counts.get(key, 0) > 0
    }


def _accumulate_metric_tensor(metric_sums, metric_counts, metrics):
    for key, value in metrics.items():
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        for item in value.detach().float().flatten().cpu().tolist():
            if math.isfinite(float(item)):
                metric_sums[key] = metric_sums.get(key, 0.0) + float(item)
                metric_counts[key] = metric_counts.get(key, 0) + 1


def _load_checkpoint(path: str):
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


def _processor_source(model_id: str, ckpt_path: str, processor_dir: Optional[str]):
    if processor_dir:
        return to_absolute_path(processor_dir)

    checkpoint_dir = Path(ckpt_path).resolve().parent
    if (checkpoint_dir / "preprocessor_config.json").is_file():
        return str(checkpoint_dir)
    return model_id


def _metadata_list(metadata: Mapping, key: str, fallback: Sequence):
    value = metadata.get(key)
    return list(value) if value else list(fallback)


def _metadata_float(metadata: Mapping, key: str, fallback: float):
    value = metadata.get(key)
    return fallback if value is None else float(value)


def _build_model(cfg: DictConfig, model_id: str, cond_dim: int):
    width = int(_cfg_select(cfg, "model.uncertainty_width", 64))
    return FrozenDepthCameraGaussian(
        model_id=model_id,
        context_dim=cond_dim,
        cache_dir=_cfg_select(cfg, "eval.hf_cache_dir", None),
        feature_channels=width,
        hidden_channels=width,
        film_hidden_dim=int(_cfg_select(cfg, "model.film_layer_width", 128)),
        max_bias=_cfg_select(cfg, "training.max_bias", None),
        min_log_variance=float(_cfg_select(cfg, "training.min_log_var", -5.0)),
        max_log_variance=float(_cfg_select(cfg, "training.max_log_var", 3.0)),
        initial_std=float(_cfg_select(cfg, "training.initial_std", 0.5)),
        variance_head_init_std=float(
            _cfg_select(cfg, "training.variance_head_init_std", 1e-3)
        ),
    )


def _build_validation_sequences(dataset, min_length: int) -> Dict[SceneKey, list[MotionSequence]]:
    grouped = defaultdict(list)
    for index, item in enumerate(dataset.items):
        grouped[
            (
                item.scene_name,
                item.scene_prefix,
                item.light,
                item.collection_speed,
                item.topology,
                item.exposure,
                item.gain,
                item.lap_id,
            )
        ].append((index, item))

    sequences_by_scene = defaultdict(list)
    for (
        scene_name,
        scene_prefix,
        light,
        speed,
        topology,
        exposure,
        gain,
        lap_id,
    ), rows in grouped.items():
        rows = sorted(rows, key=lambda pair: natural_key(pair[1].frame_id))
        if len(rows) < min_length:
            continue

        features = np.stack(
            [
                motion_features_from_metadata(
                    item.metadata_path,
                    fallback=(item.linear_speed, item.angular_speed, item.acceleration),
                )
                for _, item in rows
            ],
            axis=0,
        )
        sequence = MotionSequence(
            scene_name=scene_name,
            scene_prefix=scene_prefix,
            light=light,
            speed=speed,
            topology=topology_id(topology),
            lap_id=lap_id,
            exposure=float(exposure),
            gain=float(gain),
            features=features,
            frame_ids=[item.frame_id for _, item in rows],
            dataset_indices=[index for index, _ in rows],
        )
        sequences_by_scene[sequence.scene_key].append(sequence)

    return dict(sequences_by_scene)


def _condition_sequences(sequences: Sequence[MotionSequence]):
    grouped = defaultdict(list)
    for sequence in sequences:
        if sequence.condition_key is not None:
            grouped[sequence.condition_key].append(sequence)
    return dict(grouped)


def _choose_reference_sequence(
    sequences: Sequence[MotionSequence],
    reference_exposure: Optional[float],
    reference_gain: Optional[float],
    reference_lap: Optional[str],
) -> MotionSequence:
    candidates = list(sequences)
    if reference_exposure is not None:
        candidates = [
            sequence
            for sequence in candidates
            if sequence.exposure is not None and math.isclose(sequence.exposure, reference_exposure)
        ]
    if reference_gain is not None:
        candidates = [
            sequence
            for sequence in candidates
            if sequence.gain is not None and math.isclose(sequence.gain, reference_gain)
        ]
    if reference_lap is not None:
        candidates = [sequence for sequence in candidates if sequence.lap_id == reference_lap]

    if not candidates:
        raise ValueError("No validation sequence matches the requested reference filters.")

    return sorted(
        candidates,
        key=lambda sequence: (
            -len(sequence),
            sequence.scene_name,
            float(sequence.exposure or 0.0),
            float(sequence.gain or 0.0),
            natural_key(sequence.lap_id),
        ),
    )[0]


def _window_starts(num_frames: int, window_size: int, stride: int, drop_last: bool):
    starts = list(range(0, max(num_frames, 0), max(1, stride)))
    if drop_last:
        return [start for start in starts if start + window_size <= num_frames]
    return starts


def _dtw_cost(reference: np.ndarray, candidate: np.ndarray, feature_weights: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64) * feature_weights.reshape(1, -1)
    candidate = np.asarray(candidate, dtype=np.float64) * feature_weights.reshape(1, -1)
    if reference.size == 0 or candidate.size == 0:
        return float("inf")

    n_ref, n_candidate = reference.shape[0], candidate.shape[0]
    dp = np.full((n_ref + 1, n_candidate + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n_ref + 1):
        ref_value = reference[i - 1]
        for j in range(1, n_candidate + 1):
            diff = ref_value - candidate[j - 1]
            step_cost = float(np.sqrt(np.sum(diff * diff)))
            dp[i, j] = step_cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n_ref, n_candidate] / max(n_ref, n_candidate))


def _find_best_aligned_window(
    reference_features: np.ndarray,
    sequences: Sequence[MotionSequence],
    align_stride: int,
    feature_weights: np.ndarray,
) -> Optional[AlignedWindow]:
    window_length = int(reference_features.shape[0])
    best = None
    for sequence in sequences:
        if len(sequence) < window_length:
            continue
        for start in range(0, len(sequence) - window_length + 1, max(1, align_stride)):
            end = start + window_length
            cost = _dtw_cost(reference_features, sequence.features[start:end], feature_weights)
            if best is None or cost < best.cost:
                best = AlignedWindow(sequence=sequence, start=start, end=end, cost=cost)
    return best


def _stack_samples(dataset, indices: Sequence[int]):
    samples = [dataset[index] for index in indices]
    return (
        torch.stack([sample[0] for sample in samples], dim=0),
        torch.stack([sample[1] for sample in samples], dim=0),
        torch.stack([sample[2] for sample in samples], dim=0),
        torch.stack([sample[3] for sample in samples], dim=0),
        torch.stack([sample[4] for sample in samples], dim=0),
    )


def _prediction_depth(
    output: Mapping[str, torch.Tensor],
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    model_name: str,
    relative_align_mode: str,
) -> torch.Tensor:
    if model_name.startswith("metric"):
        return output["base_depth"]

    aligned = align_relative_prediction_to_depth_space(
        output["base_depth"],
        depth,
        valid_mask,
        align_mode=relative_align_mode,
    )
    return aligned["depth"]


@torch.no_grad()
def _evaluate_batch(
    model_name: str,
    model: FrozenDepthCameraGaussian,
    batch,
    device,
    amp: bool,
    uncertainty_mode: str,
    min_depth: float,
    max_depth: float,
    relative_align_mode: str,
):
    pixel_values, depth, valid_mask, condition, _condition_stats = batch
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

    pred_depth = _prediction_depth(
        output=output,
        depth=depth,
        valid_mask=valid_mask,
        model_name=model_name,
        relative_align_mode=relative_align_mode,
    ).detach()
    uncertainty = torch.sqrt(
        output["camera_bias"].detach().float().square()
        + output["std"].detach().float().square()
    )
    scores = image_uncertainty_score(uncertainty, valid_mask, mode=uncertainty_mode)
    metrics = compute_comprehensive_depth_metrics(
        pred_depth.float(),
        depth.float(),
        valid_mask.float(),
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return scores.detach().cpu().tolist(), metrics


def _evaluate_validation_window(
    dataset,
    indices: Sequence[int],
    model_name: str,
    model: FrozenDepthCameraGaussian,
    device,
    amp: bool,
    eval_batch_size: int,
    uncertainty_mode: str,
    min_depth: float,
    max_depth: float,
    relative_align_mode: str,
):
    uncertainty_scores = []
    metric_sums = {}
    metric_counts = {}

    for offset in range(0, len(indices), eval_batch_size):
        batch_indices = indices[offset : offset + eval_batch_size]
        scores, metrics = _evaluate_batch(
            model_name=model_name,
            model=model,
            batch=_stack_samples(dataset, batch_indices),
            device=device,
            amp=amp,
            uncertainty_mode=uncertainty_mode,
            min_depth=min_depth,
            max_depth=max_depth,
            relative_align_mode=relative_align_mode,
        )
        uncertainty_scores.extend(scores)
        _accumulate_metric_tensor(metric_sums, metric_counts, metrics)

    return {
        "uncertainty_score": _finite_mean(uncertainty_scores, default=float("inf")),
        "metrics": _mean_metric_accumulator(metric_sums, metric_counts),
        "metric_sums": metric_sums,
        "metric_counts": metric_counts,
        "num_frames": len(indices),
    }


def _evaluate_ae_window(
    ae_dataset,
    window: Optional[AlignedWindow],
    model_name: str,
    model: FrozenDepthCameraGaussian,
    device,
    amp: bool,
    eval_batch_size: int,
    uncertainty_mode: str,
    min_depth: float,
    max_depth: float,
    relative_align_mode: str,
):
    if window is None or window.sequence.dataset_indices is None:
        return {
            "valid": False,
            "metrics": {},
            "metric_sums": {},
            "metric_counts": {},
            "num_metric_frames": 0,
        }

    result = _evaluate_validation_window(
        dataset=ae_dataset,
        indices=window.sequence.dataset_indices[window.start : window.end],
        model_name=model_name,
        model=model,
        device=device,
        amp=amp,
        eval_batch_size=eval_batch_size,
        uncertainty_mode=uncertainty_mode,
        min_depth=min_depth,
        max_depth=max_depth,
        relative_align_mode=relative_align_mode,
    )
    return {
        "valid": bool(result["metric_counts"]),
        "metrics": result["metrics"],
        "metric_sums": result["metric_sums"],
        "metric_counts": result["metric_counts"],
        "num_metric_frames": result["num_frames"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _row_keys(rows: Sequence[Mapping], prefixes: Sequence[str]):
    keys = set()
    for row in rows:
        for key in row:
            if any(key.startswith(prefix) for prefix in prefixes):
                keys.add(key)
    return sorted(keys)


def _row_add_metrics(row: Dict, prefix: str, metrics: Mapping[str, float]):
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def _evaluate_aligned_scenes(
    cfg: DictConfig,
    val_dataset,
    ae_dataset,
    val_sequences_by_scene: Mapping[SceneKey, Sequence[MotionSequence]],
    ae_sequences_by_scene: Mapping[SceneKey, Sequence[MotionSequence]],
    model_name: str,
    model: FrozenDepthCameraGaussian,
    device,
    amp: bool,
):
    window_size = int(_cfg_select(cfg, "eval.window_size", 10))
    reference_stride = int(_cfg_select(cfg, "eval.reference_stride", window_size))
    align_stride = int(_cfg_select(cfg, "eval.align_stride", 1))
    drop_last = bool(_cfg_select(cfg, "eval.drop_last", True))
    max_windows = _cfg_select(cfg, "eval.max_windows_per_scene", None)
    max_windows = None if max_windows is None else int(max_windows)
    feature_weights = np.asarray(
        _cfg_list(cfg, "eval.motion_feature_weights", [1.0, 1.0, 1.0, 0.25]),
        dtype=np.float32,
    )
    if feature_weights.shape[0] != 4:
        raise ValueError("eval.motion_feature_weights must contain 4 values.")

    reference_exposure = _cfg_select(cfg, "eval.reference_exposure", None)
    reference_gain = _cfg_select(cfg, "eval.reference_gain", None)
    reference_lap = _cfg_select(cfg, "eval.reference_lap", None)
    reference_exposure = None if reference_exposure is None else float(reference_exposure)
    reference_gain = None if reference_gain is None else float(reference_gain)

    eval_batch_size = int(_cfg_select(cfg, "eval.eval_batch_size", _cfg_select(cfg, "training.batch_size", 4)))
    uncertainty_mode = str(_cfg_select(cfg, "training.uncertainty_mode", "mean"))
    min_depth = float(_cfg_select(cfg, "dataset.min_depth", 1e-3))
    max_depth = float(_cfg_select(cfg, "dataset.max_depth", 10.0))
    relative_align_mode = str(_cfg_select(cfg, "training.relative_align_mode", "scale_shift"))

    window_rows = []
    candidate_rows = []
    scene_accumulators = {}

    for scene_key in sorted(val_sequences_by_scene, key=lambda key: (key.light, key.speed, key.topology)):
        validation_sequences = list(val_sequences_by_scene[scene_key])
        candidates_by_condition = _condition_sequences(validation_sequences)
        if not candidates_by_condition:
            continue

        reference = _choose_reference_sequence(
            validation_sequences,
            reference_exposure=reference_exposure,
            reference_gain=reference_gain,
            reference_lap=reference_lap,
        )
        starts = _window_starts(
            len(reference),
            window_size=window_size,
            stride=reference_stride,
            drop_last=drop_last,
        )
        if max_windows is not None:
            starts = starts[:max_windows]
        if not starts:
            print(f"[skip] {reference.scene_name}: no reference windows")
            continue

        ae_sequences = list(ae_sequences_by_scene.get(scene_key, []))
        scene_accumulators[scene_key] = {
            "reference_scene": reference.scene_name,
            "validation_sums": {},
            "validation_counts": {},
            "ae_sums": {},
            "ae_counts": {},
            "windows": 0,
            "ae_windows": 0,
            "frames": 0,
            "selection_counts": defaultdict(int),
            "selected_uncertainties": [],
            "alignment_costs": [],
            "ae_alignment_costs": [],
        }

        print(
            f"[scene] {scene_key.light}_{scene_key.speed}_{scene_key.topology}: "
            f"{len(candidates_by_condition)} validation conditions, "
            f"{len(validation_sequences)} validation laps, "
            f"{len(ae_sequences)} AE laps, {len(starts)} windows"
        )

        for window_index, start in enumerate(starts):
            end = min(start + window_size, len(reference))
            reference_features = reference.features[start:end]
            if reference_features.shape[0] == 0:
                continue

            candidate_results = []
            for condition, condition_sequences in sorted(candidates_by_condition.items()):
                aligned = _find_best_aligned_window(
                    reference_features=reference_features,
                    sequences=condition_sequences,
                    align_stride=align_stride,
                    feature_weights=feature_weights,
                )
                if aligned is None or aligned.sequence.dataset_indices is None:
                    continue

                result = _evaluate_validation_window(
                    dataset=val_dataset,
                    indices=aligned.sequence.dataset_indices[aligned.start : aligned.end],
                    model_name=model_name,
                    model=model,
                    device=device,
                    amp=amp,
                    eval_batch_size=eval_batch_size,
                    uncertainty_mode=uncertainty_mode,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    relative_align_mode=relative_align_mode,
                )
                candidate_results.append(
                    {
                        "condition": condition,
                        "window": aligned,
                        **result,
                    }
                )

            finite_candidates = [
                result
                for result in candidate_results
                if math.isfinite(float(result["uncertainty_score"]))
            ]
            if not finite_candidates:
                print(f"  [skip] window {window_index:04d}: no finite validation uncertainty")
                continue

            selected = min(
                finite_candidates,
                key=lambda result: (
                    result["uncertainty_score"],
                    result["window"].cost,
                    result["condition"][0],
                    result["condition"][1],
                ),
            )
            selected_window = selected["window"]
            selected_exposure, selected_gain = selected["condition"]
            selected_key = _format_condition_key(selected_exposure, selected_gain)

            ae_window = _find_best_aligned_window(
                reference_features=reference_features,
                sequences=ae_sequences,
                align_stride=align_stride,
                feature_weights=feature_weights,
            )
            ae_result = _evaluate_ae_window(
                ae_dataset=ae_dataset,
                window=ae_window,
                model_name=model_name,
                model=model,
                device=device,
                amp=amp,
                eval_batch_size=eval_batch_size,
                uncertainty_mode=uncertainty_mode,
                min_depth=min_depth,
                max_depth=max_depth,
                relative_align_mode=relative_align_mode,
            )

            acc = scene_accumulators[scene_key]
            acc["windows"] += 1
            acc["frames"] += selected["num_frames"]
            acc["selection_counts"][selected_key] += 1
            acc["selected_uncertainties"].append(selected["uncertainty_score"])
            acc["alignment_costs"].append(selected_window.cost)
            _merge_metric_accumulators(
                acc["validation_sums"],
                acc["validation_counts"],
                selected["metric_sums"],
                selected["metric_counts"],
            )
            if ae_result["valid"]:
                acc["ae_windows"] += 1
                acc["ae_alignment_costs"].append(ae_window.cost)
                _merge_metric_accumulators(
                    acc["ae_sums"],
                    acc["ae_counts"],
                    ae_result["metric_sums"],
                    ae_result["metric_counts"],
                )

            row = {
                "scene_light": scene_key.light,
                "scene_speed": scene_key.speed,
                "scene_topology": scene_key.topology,
                "reference_scene": reference.scene_name,
                "reference_lap": reference.lap_id,
                "reference_exposure": reference.exposure,
                "reference_gain": reference.gain,
                "window_index": window_index,
                "reference_frame_start": reference.frame_ids[start],
                "reference_frame_end": reference.frame_ids[end - 1],
                "num_frames": selected["num_frames"],
                "selected_scene": selected_window.sequence.scene_name,
                "selected_lap": selected_window.sequence.lap_id,
                "selected_frame_start": selected_window.frame_ids[0],
                "selected_frame_end": selected_window.frame_ids[-1],
                "selected_exposure": selected_exposure,
                "selected_gain": selected_gain,
                "selected_condition": selected_key,
                "selected_uncertainty": selected["uncertainty_score"],
                "selection_align_cost": selected_window.cost,
                "ae_valid": int(ae_result["valid"]),
                "ae_scene": ae_window.sequence.scene_name if ae_window is not None else "",
                "ae_lap": ae_window.sequence.lap_id if ae_window is not None else "",
                "ae_frame_start": ae_window.frame_ids[0] if ae_window is not None else "",
                "ae_frame_end": ae_window.frame_ids[-1] if ae_window is not None else "",
                "ae_align_cost": ae_window.cost if ae_window is not None else float("nan"),
                "ae_num_metric_frames": ae_result["num_metric_frames"],
            }
            _row_add_metrics(row, "validation_metric", selected["metrics"])
            _row_add_metrics(row, "ae_metric", ae_result["metrics"])
            if "abs_rel" in selected["metrics"] and "abs_rel" in ae_result["metrics"]:
                row["delta_abs_rel_validation_minus_ae"] = (
                    selected["metrics"]["abs_rel"] - ae_result["metrics"]["abs_rel"]
                )
            if "a1" in selected["metrics"] and "a1" in ae_result["metrics"]:
                row["delta_a1_validation_minus_ae"] = (
                    selected["metrics"]["a1"] - ae_result["metrics"]["a1"]
                )
            window_rows.append(row)

            for candidate in candidate_results:
                candidate_window = candidate["window"]
                exposure, gain = candidate["condition"]
                candidate_row = {
                    "scene_light": scene_key.light,
                    "scene_speed": scene_key.speed,
                    "scene_topology": scene_key.topology,
                    "reference_scene": reference.scene_name,
                    "reference_lap": reference.lap_id,
                    "window_index": window_index,
                    "reference_frame_start": reference.frame_ids[start],
                    "reference_frame_end": reference.frame_ids[end - 1],
                    "candidate_scene": candidate_window.sequence.scene_name,
                    "candidate_lap": candidate_window.sequence.lap_id,
                    "candidate_frame_start": candidate_window.frame_ids[0],
                    "candidate_frame_end": candidate_window.frame_ids[-1],
                    "exposure": exposure,
                    "gain": gain,
                    "condition": _format_condition_key(exposure, gain),
                    "num_frames": candidate["num_frames"],
                    "align_cost": candidate_window.cost,
                    "uncertainty_score": candidate["uncertainty_score"],
                    "selected": int(candidate is selected),
                }
                _row_add_metrics(candidate_row, "metric", candidate["metrics"])
                candidate_rows.append(candidate_row)

            validation_text = ", ".join(
                f"{key}={value:.4f}"
                for key, value in selected["metrics"].items()
                if key in ("abs_rel", "a1")
            )
            ae_text = ", ".join(
                f"{key}={value:.4f}"
                for key, value in ae_result["metrics"].items()
                if key in ("abs_rel", "a1")
            )
            print(
                f"  window {window_index:04d}: selected {selected_key} "
                f"unc={selected['uncertainty_score']:.6f} "
                f"align={selected_window.cost:.4f} "
                f"val[{validation_text}] ae[{ae_text or 'missing metrics'}]"
            )

    scene_rows = []
    for scene_key, acc in sorted(
        scene_accumulators.items(),
        key=lambda item: (item[0].light, item[0].speed, item[0].topology),
    ):
        validation_metrics = _mean_metric_accumulator(
            acc["validation_sums"],
            acc["validation_counts"],
        )
        ae_metrics = _mean_metric_accumulator(acc["ae_sums"], acc["ae_counts"])
        row = {
            "scene_light": scene_key.light,
            "scene_speed": scene_key.speed,
            "scene_topology": scene_key.topology,
            "reference_scene": acc["reference_scene"],
            "windows": acc["windows"],
            "ae_windows": acc["ae_windows"],
            "frames": acc["frames"],
            "mean_selected_uncertainty": _finite_mean(acc["selected_uncertainties"]),
            "mean_selection_align_cost": _finite_mean(acc["alignment_costs"]),
            "mean_ae_align_cost": _finite_mean(acc["ae_alignment_costs"]),
            "selection_counts": json.dumps(dict(sorted(acc["selection_counts"].items()))),
        }
        _row_add_metrics(row, "validation_metric", validation_metrics)
        _row_add_metrics(row, "ae_metric", ae_metrics)
        if "abs_rel" in validation_metrics and "abs_rel" in ae_metrics:
            row["delta_abs_rel_validation_minus_ae"] = (
                validation_metrics["abs_rel"] - ae_metrics["abs_rel"]
            )
        if "a1" in validation_metrics and "a1" in ae_metrics:
            row["delta_a1_validation_minus_ae"] = (
                validation_metrics["a1"] - ae_metrics["a1"]
            )
        scene_rows.append(row)

    return window_rows, candidate_rows, scene_rows


@hydra.main(version_base=None, config_path="config", config_name="base_mdebias")
def main(cfg: DictConfig):
    window_size = int(_cfg_select(cfg, "eval.window_size", 10))
    if window_size <= 0:
        raise ValueError("eval.window_size must be positive.")

    seed_everything(int(_cfg_select(cfg, "training.seed", 2026)))

    ckpt_path = _cfg_select(cfg, "eval.ckpt", None)
    if ckpt_path is None:
        raise ValueError("Set eval.ckpt=/path/to/checkpoint.pt")
    ckpt_path = to_absolute_path(str(ckpt_path))
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    checkpoint, state_dict, metadata = _load_checkpoint(ckpt_path)
    model_name = str(_cfg_select(cfg, "model.model_id", "small"))
    model_id = MODEL_IDS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") and (not bool(_cfg_select(cfg, "training.no_amp", False)))

    processor_dir = _cfg_select(cfg, "eval.processor_dir", None)
    image_processor = AutoImageProcessor.from_pretrained(
        _processor_source(model_id, ckpt_path, processor_dir),
        cache_dir=_cfg_select(cfg, "eval.hf_cache_dir", None),
    )

    dataset_root = to_absolute_path(
        str(_cfg_select(cfg, "dataset.dataset_root", DEFAULT_DATASET_ROOT))
    )
    ae_root = to_absolute_path(str(_cfg_select(cfg, "eval.ae_root", dataset_root)))
    min_depth = float(_cfg_select(cfg, "dataset.min_depth", 1e-3))
    max_depth = float(_cfg_select(cfg, "dataset.max_depth", 10.0))
    min_valid_depth_ratio = float(_cfg_select(cfg, "dataset.min_valid_depth_ratio", 0.0))
    light_levels = _metadata_list(metadata, "light_levels", LIGHT_LEVELS)
    speed_levels = _metadata_list(metadata, "speed_levels", MOTION_LEVELS)
    image_size = (
        int(_cfg_select(cfg, "model.image_height", 518)),
        int(_cfg_select(cfg, "model.image_width", 518)),
    )

    dataset_kwargs = {
        "root_dir": dataset_root,
        "image_processor": image_processor,
        "image_size": image_size,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "min_valid_depth_ratio": min_valid_depth_ratio,
        "light_levels": light_levels,
        "speed_levels": speed_levels,
    }

    val_dataset = ATIRealWorldUncertaintyValidationDataset(**dataset_kwargs)
    if all(key in metadata for key in ("exposure_min", "exposure_max", "gain_min", "gain_max")):
        val_dataset.exposure_min = _metadata_float(metadata, "exposure_min", val_dataset.exposure_min)
        val_dataset.exposure_max = _metadata_float(metadata, "exposure_max", val_dataset.exposure_max)
        val_dataset.gain_min = _metadata_float(metadata, "gain_min", val_dataset.gain_min)
        val_dataset.gain_max = _metadata_float(metadata, "gain_max", val_dataset.gain_max)
    else:
        train_topologies = _cfg_select(cfg, "dataset.train_topologies", None)
        if train_topologies:
            train_dataset = ATIRealWorldUncertaintyDataset(
                topologies=train_topologies,
                **dataset_kwargs,
            )
            copy_condition_normalization(val_dataset, train_dataset)

    cond_dim = len(metadata.get("condition_names", val_dataset.condition_names))
    if cond_dim != val_dataset.condition_dim:
        raise ValueError(
            f"Checkpoint condition dim ({cond_dim}) does not match validation "
            f"dataset condition dim ({val_dataset.condition_dim})."
        )

    model = _build_model(cfg, model_id=model_id, cond_dim=cond_dim).to(device)
    load_result = model.load_state_dict(
        state_dict,
        strict=not bool(_cfg_select(cfg, "eval.non_strict_load", False)),
    )
    model.eval()

    ae_dataset = AutoExposureMotionDataset(
        root_dir=ae_root,
        image_processor=image_processor,
        image_size=image_size,
        min_depth=min_depth,
        max_depth=max_depth,
        min_valid_depth_ratio=min_valid_depth_ratio,
        light_levels=light_levels,
        speed_levels=speed_levels,
        min_length=window_size,
        ae_exposure=_cfg_select(cfg, "eval.ae_exposure", None),
        ae_gain=_cfg_select(cfg, "eval.ae_gain", None),
    )
    ae_dataset.exposure_min = val_dataset.exposure_min
    ae_dataset.exposure_max = val_dataset.exposure_max
    ae_dataset.gain_min = val_dataset.gain_min
    ae_dataset.gain_max = val_dataset.gain_max

    output_dir = Path(str(_cfg_select(cfg, "eval.output_dir", "./unc_selection_results")))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {model_id}")
    print("model class: FrozenDepthCameraGaussian")
    print(f"checkpoint: {ckpt_path}")
    print(f"dataset root: {dataset_root}")
    print(f"AE root: {ae_root}")
    print(f"validation samples: {len(val_dataset):,}")
    print(f"AE frames: {len(ae_dataset):,}")
    print(f"AE sequences: {len(ae_dataset.sequences):,}")
    print(f"AE scan stats: {ae_dataset.scan_stats}")
    print(f"condition names: {list(val_dataset.condition_names)}")
    if bool(_cfg_select(cfg, "eval.non_strict_load", False)):
        print(f"load_state_dict result: {load_result}")

    val_sequences_by_scene = _build_validation_sequences(
        val_dataset,
        min_length=window_size,
    )
    ae_sequences_by_scene = ae_dataset.sequences_by_scene()

    print(
        f"validation aligned scenes: {len(val_sequences_by_scene)}, "
        f"AE aligned scenes: {len(ae_sequences_by_scene)}"
    )

    window_rows, candidate_rows, scene_rows = _evaluate_aligned_scenes(
        cfg=cfg,
        val_dataset=val_dataset,
        ae_dataset=ae_dataset,
        val_sequences_by_scene=val_sequences_by_scene,
        ae_sequences_by_scene=ae_sequences_by_scene,
        model_name=model_id,
        model=model,
        device=device,
        amp=amp,
    )

    window_fields = [
        "scene_light",
        "scene_speed",
        "scene_topology",
        "reference_scene",
        "reference_lap",
        "reference_exposure",
        "reference_gain",
        "window_index",
        "reference_frame_start",
        "reference_frame_end",
        "num_frames",
        "selected_scene",
        "selected_lap",
        "selected_frame_start",
        "selected_frame_end",
        "selected_exposure",
        "selected_gain",
        "selected_condition",
        "selected_uncertainty",
        "selection_align_cost",
        "ae_valid",
        "ae_scene",
        "ae_lap",
        "ae_frame_start",
        "ae_frame_end",
        "ae_align_cost",
        "ae_num_metric_frames",
        *_row_keys(window_rows, ("validation_metric_", "ae_metric_", "delta_")),
    ]
    candidate_fields = [
        "scene_light",
        "scene_speed",
        "scene_topology",
        "reference_scene",
        "reference_lap",
        "window_index",
        "reference_frame_start",
        "reference_frame_end",
        "candidate_scene",
        "candidate_lap",
        "candidate_frame_start",
        "candidate_frame_end",
        "exposure",
        "gain",
        "condition",
        "num_frames",
        "align_cost",
        "uncertainty_score",
        "selected",
        *_row_keys(candidate_rows, ("metric_",)),
    ]
    scene_fields = [
        "scene_light",
        "scene_speed",
        "scene_topology",
        "reference_scene",
        "windows",
        "ae_windows",
        "frames",
        "mean_selected_uncertainty",
        "mean_selection_align_cost",
        "mean_ae_align_cost",
        "selection_counts",
        *_row_keys(scene_rows, ("validation_metric_", "ae_metric_", "delta_")),
    ]

    _write_csv(output_dir / "unc_selection_window_results.csv", window_rows, window_fields)
    _write_csv(output_dir / "unc_selection_candidate_results.csv", candidate_rows, candidate_fields)
    _write_csv(output_dir / "scene_summary.csv", scene_rows, scene_fields)

    summary_payload = {
        "args": OmegaConf.to_container(cfg, resolve=True),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_val_metrics": checkpoint.get("val_metrics") if isinstance(checkpoint, dict) else None,
        "dataset_metadata": metadata,
        "ae_scan_stats": ae_dataset.scan_stats,
        "num_window_rows": len(window_rows),
        "num_candidate_rows": len(candidate_rows),
        "num_scene_rows": len(scene_rows),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"saved window results: {output_dir / 'unc_selection_window_results.csv'}")
    print(f"saved candidate results: {output_dir / 'unc_selection_candidate_results.csv'}")
    print(f"saved scene summary: {output_dir / 'scene_summary.csv'}")
    print(f"saved summary: {output_dir / 'summary.json'}")
    if scene_rows:
        print("[scene report]")
        for row in scene_rows:
            val_abs_rel = row.get("validation_metric_abs_rel", float("nan"))
            val_a1 = row.get("validation_metric_a1", float("nan"))
            ae_abs_rel = row.get("ae_metric_abs_rel", float("nan"))
            ae_a1 = row.get("ae_metric_a1", float("nan"))
            print(
                f"  {row['scene_light']}/{row['scene_speed']}/{row['scene_topology']}: "
                f"val abs_rel={val_abs_rel:.4f}, val a1={val_a1:.4f}, "
                f"AE abs_rel={ae_abs_rel:.4f}, AE a1={ae_a1:.4f}, "
                f"windows={row['windows']}, ae_windows={row['ae_windows']}"
            )


if __name__ == "__main__":
    main()
