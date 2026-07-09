#!/usr/bin/env python3
# Auto-generated from ATIMDE-CanonicalParams.ipynb.
# Run from any directory with: python ATIMDE-CanonicalParams.py

from pathlib import Path as _Path
import os as _os

from evaluation_utils.eval_utils import masked_median
_SCRIPT_DIR = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_os.chdir(_SCRIPT_DIR)
del _Path, _os

# %% Cell 0
from pathlib import Path
import gc
import json
import math
import os
import re
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from itertools import product
from typing import Dict, Tuple, Optional

# %% Cell 1
V_STOP = 0.03
W_STOP = 0.05
V_SLOW = 0.25
V_FAST = 0.75
W_ROTATE = 0.5
W_SPIN = 1.0
A_SLOW = 0.5
A_FAST = 1.0

MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "metric-indoor-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-indoor-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "metric-outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "metric-outdoor-base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
}
MODEL_ID = MODEL_IDS["small"]
HF_CACHE_DIR = os.environ.get("HF_HOME") or None
LOCAL_FILES_ONLY = os.environ.get("DEPTH_ANYTHING_LOCAL_FILES_ONLY", "1") != "0"
EVAL_BATCH_SIZE = int(os.environ.get("DEPTH_EVAL_BATCH_SIZE", "16"))


def assign_motion_label(
    v, a, w,
    v_stop=V_STOP, w_stop=W_STOP, w_rotate=W_ROTATE, w_spin=W_SPIN,
):
    """
    v: wheel linear velocity magnitude
    a: IMU acceleration magnitude
    w: abs wheel angular velocity yaw_z

    Label priority:
      1. STOP
      2. SPIN / ROTATE
      3. FAST / NORMAL / SLOW
    """
    if np.isnan(v):
        v = 0.0
    if np.isnan(a):
        a = 0.0
    if np.isnan(w):
        w = 0.0
    # true stop
    if v < v_stop and w < w_stop:
        return "stop"
    # rotation-dominant
    if w >= w_spin and v < 0.25:
        return "spin"
    if w >= w_rotate and v < 0.30:
        return "rotate"
    # translation-dominant
    if a >= A_SLOW and w < w_rotate:
        return "fast"
    elif w < w_rotate:
        return "slow"
    elif w >= w_rotate:
        return "rotate"
    return "stop"

def abs_float(x):
    try:
        return abs(float(x))
    except Exception:
        return np.nan

def xy_mag(x, y):
    try:
        return math.sqrt(float(x) ** 2 + float(y) ** 2)
    except Exception:
        return np.nan

def safe_get(d, keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
        
def get_motion_data(json_data):
    try:
        imu_ax = safe_get(json_data, ["imu", "linear_acceleration", "x"])
        imu_ay = safe_get(json_data, ["imu", "linear_acceleration", "y"])
        wheel_vx = safe_get(json_data, ["wheel_odometry", "linear_velocity", "x"])
        wheel_vy = safe_get(json_data, ["wheel_odometry", "linear_velocity", "y"])
        wheel_yaw = safe_get(json_data, ["wheel_odometry", "angular_velocity", "yaw_z"])

        accel_mag = xy_mag(imu_ax, imu_ay)
        lin_speed = xy_mag(wheel_vx, wheel_vy)
        return accel_mag, lin_speed, abs_float(wheel_yaw)
    except Exception as e:
        print(e)
        return np.nan, np.nan, np.nan

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_counts = mask.flatten(1).sum(dim=1)
    sums = torch.where(mask, values, torch.zeros_like(values)).flatten(1).sum(dim=1)
    means = sums / valid_counts.clamp_min(1).to(dtype=values.dtype)
    return torch.where(valid_counts > 0, means, torch.full_like(means, float("nan")))
    
def load_depth_anything_metric_indoor(device):
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        local_files_only=LOCAL_FILES_ONLY,
    ).eval().to(device)
    return model, processor

def predict_metric_depth_batch(model, processor, rgb_bgr_list, target_shapes, device):
    rgb_images = [Image.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)) for rgb_bgr in rgb_bgr_list]
    inputs = processor(images=rgb_images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    processed = processor.post_process_depth_estimation(outputs, target_sizes=target_shapes)
    return [entry["predicted_depth"].to(device=device, dtype=torch.float32) for entry in processed]

def predict_metric_depth(model, processor, rgb_bgr, target_shape, device):
    processed = predict_metric_depth_batch(model, processor, [rgb_bgr], [target_shape], device)
    return processed[0].detach().cpu().numpy().astype(np.float32)

def frame_time_sec(frame_metadata):
    capture_time = frame_metadata.get("capture_time", {}) or {}
    if capture_time.get("time_sec") is not None:
        return float(capture_time["time_sec"])
    if capture_time.get("sec") is not None and capture_time.get("nanosec") is not None:
        return float(capture_time["sec"]) + float(capture_time["nanosec"]) * 1e-9

    camera = frame_metadata.get("camera", {}) or {}
    color_frame = camera.get("color_frame", {}) or {}
    if color_frame.get("timestamp_ms") is not None:
        return float(color_frame["timestamp_ms"]) * 1e-3

    raise KeyError("No supported timestamp found in frame metadata")

def lap_duration_sec(lap_dir):
    metadata_paths = sorted((Path(lap_dir) / "metadata").glob("*.json"))
    if len(metadata_paths) < 2:
        return None
    first_metadata = json.load(open(metadata_paths[0], "r"))
    last_metadata = json.load(open(metadata_paths[-1], "r"))
    return frame_time_sec(last_metadata), frame_time_sec(first_metadata)

# %% Cell 2
MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0

def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected [B, H, W] or [B, 1, H, W], got {tuple(x.shape)}")

def metric_dict(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    min_depth=MIN_DEPTH,
    max_depth=MAX_DEPTH,
    eps=1e-8,
    calc_dtype=torch.float32,
) -> Dict[str, torch.Tensor]:
    pred_depth = torch.clamp(pred_depth, min_depth, max_depth)
    gt_depth = torch.clamp(gt_depth, min_depth, max_depth)
    mask = mask & torch.isfinite(pred_depth) & torch.isfinite(gt_depth)

    diff = gt_depth - pred_depth
    sq_error = diff.square()
    thresh = torch.maximum(gt_depth / (pred_depth + eps), pred_depth / (gt_depth + eps))
    log_error = (torch.log(gt_depth + eps) - torch.log(pred_depth + eps)).square()

    return {
        "abs_rel": masked_mean(torch.abs(diff) / (gt_depth + eps), mask),
        "sq_rel": masked_mean(sq_error / (gt_depth + eps), mask),
        "rmse": torch.sqrt(masked_mean(sq_error, mask)),
        "rmse_log": torch.sqrt(masked_mean(log_error, mask)),
        "a1": masked_mean((thresh < 1.25).to(dtype=calc_dtype), mask),
        "a2": masked_mean((thresh < 1.25 ** 2).to(dtype=calc_dtype), mask),
        "a3": masked_mean((thresh < 1.25 ** 3).to(dtype=calc_dtype), mask),
    }

@torch.no_grad()
def compute_relative_alignment(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    align_mode: str = "scale_shift",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = ensure_bchw(pred)
    gt = ensure_bchw(gt)
    valid_mask = ensure_bchw(valid_mask).bool()
    calc_dtype = torch.float64 if pred.dtype == torch.float64 or gt.dtype == torch.float64 else torch.float32
    pred = pred.to(dtype=calc_dtype)
    gt = gt.to(dtype=calc_dtype)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: target {gt.shape}, pred {pred.shape}")
    if valid_mask.shape != pred.shape:
        valid_mask = valid_mask.expand_as(pred)

    relative_mask = valid_mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0) & (pred > 0)
    gt_inv = 1.0 / (gt + eps)

    if align_mode == "median":
        pred_median = masked_median(pred, relative_mask)
        gt_inv_median = masked_median(gt_inv, relative_mask)
        scale = gt_inv_median / (pred_median + eps)
        shift = torch.zeros_like(scale)
    elif align_mode == "scale_shift":
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
    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")

    return scale.view(-1, 1, 1, 1), shift.view(-1, 1, 1, 1)

def align_relative_prediction_to_depth_space(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    align_mode: str = "scale_shift",
    inv_depth_min: float = 1e-1,
    eps: float = 1e-8,
) -> Dict[str, Optional[torch.Tensor]]:
    pred = ensure_bchw(pred)
    gt = ensure_bchw(gt)
    calc_dtype = torch.float64 if pred.dtype == torch.float64 or gt.dtype == torch.float64 else torch.float32
    pred = pred.to(dtype=calc_dtype)

    scale, shift = compute_relative_alignment(
        pred,
        gt,
        valid_mask,
        align_mode=align_mode,
        eps=eps,
    )
    aligned_inv_depth = scale * pred + shift
    safe_inv_depth = aligned_inv_depth.clamp_min(inv_depth_min)
    pred_depth = 1.0 / safe_inv_depth

    result = {
        "depth": pred_depth,
        "aligned_inv_depth": safe_inv_depth,
        "scale": scale,
        "shift": shift,
    }
    return result

@torch.no_grad()
def compute_comprehensive_depth_metrics(
    mu: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    min_depth=MIN_DEPTH,
    max_depth=MAX_DEPTH,
) -> Dict[str, torch.Tensor]:
    """
    Calculating depth metrics in parallel manner with CUDA GPU and PyTorch. 
    This function computes both absolute depth metrics and relative depth metrics for a batch of images.
    Returning Dictionary of tensors of each metrics. (Abs_rel, RMSE, A1, A2, A3)
    
        Input: 
            - mu: Predicted depth map tensor of shape [B, 1, H, W]
            - target: Ground truth depth map tensor of shape [B, 1, H, W]
            - valid_mask: Boolean tensor indicating valid pixels of shape [B, 1, H, W]
            - min_depth: Minimum depth value for clamping
            - max_depth: Maximum depth value for clamping 
    
        Output:
            - metrics: Dictionary containing computed depth metrics 
            Key: "abs_rel", "rmse", "sq_rel", "a1", "a2", "a3"
            Values: Tensor of shape [B] containing the metric for each image in the batch
    """
    pred = ensure_bchw(mu.detach())
    gt = ensure_bchw(target.detach())
    valid_mask = ensure_bchw(valid_mask).bool()
    calc_dtype = torch.float64 if mu.dtype == torch.float64 or target.dtype == torch.float64 else torch.float32
    pred = pred.to(dtype=calc_dtype)
    gt = gt.to(dtype=calc_dtype)
    if pred.shape != gt.shape:
        raise ValueError(...)
    if valid_mask.shape != pred.shape:
        valid_mask = valid_mask.expand_as(pred)
    
    eps = 1e-8
    valid_mask = valid_mask.bool()
    metric_mask = valid_mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0)
    return metric_dict(
        pred, 
        gt, 
        metric_mask, 
        min_depth=min_depth, 
        max_depth=max_depth, 
        eps=eps, 
        calc_dtype=calc_dtype
    )

# %% Cell 3
from bisect import bisect_left
from functools import lru_cache
import csv
import math

CANONICAL_MATCH_OUTPUT_CSV = Path("canonical_parameter_frame_matches.csv")
CANONICAL_MATCH_OUTPUT_DIR = Path("orbbec_canonical_parameter_frame_matches_by_scene")
CANONICAL_TOPK_TIME_CANDIDATES = 5
RGB_MATCH_PATCH_SIZE = 256
DTW_DEPTH_FEATURE_SIZE = (8, 8)
DTW_MAX_SEQUENCE_LENGTH = 120
DEPTH_MATCH_MAX_MEAN_ABS_DIFF = 0.05 # 5cm
DEPTH_AUTO_SCALE_THRESHOLD = 20.0
DEPTH_RAW_TO_METER_SCALE = 0.001
REGISTRATION_IMAGE_SIZE = (160, 120)
REGISTRATION_MIN_OVERLAP_RATIO = 0.95
REGISTRATION_ECC_MAX_ITERS = 60
REGISTRATION_ECC_EPS = 1e-4
DEPTH_MATCH_MIN_ECC_SCORE = 0.9
REGISTRATION_SCORE_TIME_WEIGHT = 0.02
NO_MATCH_VALUE = -1

def _pair_dir_by_exposure_gain():
    return {
        (int(e), int(g)): f"pair_{i:03d}_{p_prefix}"
        for i, (e, g, p_prefix) in enumerate(PARAM_PAIRS)
    }


def _pair_index_by_exposure_gain():
    return {
        (int(e), int(g)): int(i)
        for i, (e, g, _p_prefix) in enumerate(PARAM_PAIRS)
    }

CSV_COLUMNS = [
    "scene",
    "source_pair_index",
    "source_pair_dir",
    "source_exposure",
    "source_gain",
    "source_lap_dir",
    "source_frame_index",
    "source_time_sec",
    "source_motion_label",
    "source_rgb_path",
    "source_depth_path",
    "canonical_exposure",
    "canonical_gain",
    "canonical_pair_index",
    "canonical_pair_dir",
    "matched_lap_dir",
    "matched_frame_index",
    "matched_time_sec",
    "time_diff_sec",
    "matched_lap_dtw_cost",
    "raw_depth_mean_abs_diff",
    "depth_mean_abs_diff",
    "depth_diff_threshold",
    "registered_depth_mean_abs_diff",
    "registered_depth_rmse_diff",
    "registered_depth_max_abs_diff",
    "registration_overlap_ratio",
    "registration_ecc_score",
    "registration_dx_px",
    "registration_dy_px",
    "registration_status",
    "rgb_patch_x0",
    "rgb_patch_y0",
    "rgb_patch_width",
    "rgb_patch_height",
    "rgb_patch_mean_abs_diff",
    "rgb_patch_rmse_diff",
    "rgb_patch_max_abs_diff",
    "rgb_mean_abs_diff",
    "rgb_rmse_diff",
    "rgb_max_abs_diff",
    "matched_rgb_path",
    "matched_depth_path",
    "match_status",
]


def _canonical_param_tuple(param_values):
    try:
        exposure = int(param_values[0])
        gain = int(param_values[1])
    except (TypeError, ValueError, IndexError):
        return None

    pair_key = (exposure, gain)
    pair_dir = _pair_dir_by_exposure_gain().get(pair_key)
    pair_index = _pair_index_by_exposure_gain().get(pair_key)
    if exposure <= 0 or gain <= 0 or pair_dir is None or pair_index is None:
        return None
    return exposure, gain, pair_index, pair_dir


def _load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def _metadata_paths(lap_dir):
    return sorted((Path(lap_dir) / "metadata").glob("*.json"))


def _lap_start_time_sec(lap_dir):
    metadata_paths = _metadata_paths(lap_dir)
    if not metadata_paths:
        return None
    return frame_time_sec(_load_json(metadata_paths[0]))


def _motion_label_from_metadata(metadata_json):
    accel, linear_vel, yaw_ang = get_motion_data(metadata_json)
    return assign_motion_label(linear_vel, accel, yaw_ang)


def _iter_frame_records(scene_name, pair_index, exposure, gain, pair_dir, lap_dir):
    lap_dir = Path(lap_dir)
    lap_start_time = _lap_start_time_sec(lap_dir)
    if lap_start_time is None:
        return

    for metadata_path in _metadata_paths(lap_dir):
        try:
            frame_index = int(metadata_path.stem)
            metadata_json = _load_json(metadata_path)
            curr_time = frame_time_sec(metadata_json) - lap_start_time
            motion_label = _motion_label_from_metadata(metadata_json)
        except Exception as exc:
            print(f"Skip unreadable metadata {metadata_path}: {exc}")
            continue

        yield {
            "scene": scene_name,
            "pair_index": int(pair_index),
            "pair_dir": str(pair_dir),
            "exposure": int(exposure),
            "gain": int(gain),
            "lap_dir_name": lap_dir.name,
            "frame_index": int(frame_index),
            "time_sec": float(curr_time),
            "motion_label": motion_label,
            "rgb_path": str(lap_dir / "rgb" / f"{frame_index:06d}.png"),
            "depth_path": str(lap_dir / "depth" / f"{frame_index:06d}.npy"),
        }


def _scan_pair_records(scene_name, exposure, gain, pair_index, pair_dir):
    pair_path = Path(DATA_PATH) / scene_name / pair_dir
    records = []
    for lap_dir in sorted(pair_path.glob("lap_*")):
        records.extend(
            _iter_frame_records(scene_name, pair_index, exposure, gain, pair_dir, lap_dir)
        )
    return records


pair_record_cache = {}


def _get_pair_records(scene_name, exposure, gain, pair_index, pair_dir):
    cache_key = (scene_name, int(exposure), int(gain))
    if cache_key not in pair_record_cache:
        pair_record_cache[cache_key] = _scan_pair_records(
            scene_name, exposure, gain, pair_index, pair_dir
        )
    return pair_record_cache[cache_key]


def _build_canonical_frame_index(scene_names=None):
    if scene_names is None:
        scene_names = list(canonical_parameter.keys())
    elif isinstance(scene_names, str):
        scene_names = [scene_names]

    index = {
        scene_name: {motion_label: {"times": [], "records": [], "laps": []} for motion_label in MOTION_SET}
        for scene_name in scene_names
    }

    for scene_name in scene_names:
        motion_params = canonical_parameter.get(scene_name, {})
        for motion_label in MOTION_SET:
            canonical_param = _canonical_param_tuple(motion_params.get(motion_label))
            if canonical_param is None:
                continue

            exposure, gain, pair_index, pair_dir = canonical_param
            pair_records = _get_pair_records(scene_name, exposure, gain, pair_index, pair_dir)
            pair_records = [
                record for record in pair_records
                if Path(record["rgb_path"]).exists() and Path(record["depth_path"]).exists()
            ]
            records_by_lap = {}
            for record in pair_records:
                records_by_lap.setdefault(record["lap_dir_name"], []).append(record)

            lap_infos = []
            for lap_dir_name, lap_records in records_by_lap.items():
                lap_records.sort(key=lambda record: record["time_sec"])
                motion_records_by_label = {
                    m: [record for record in lap_records if record["motion_label"] == m]
                    for m in MOTION_SET
                }
                lap_infos.append({
                    "lap_dir_name": lap_dir_name,
                    "records": lap_records,
                    "times": [record["time_sec"] for record in lap_records],
                    "motion_records": motion_records_by_label,
                    "motion_times": {
                        m: [record["time_sec"] for record in records]
                        for m, records in motion_records_by_label.items()
                    },
                })

            motion_records = [
                record for record in pair_records
                if record["motion_label"] == motion_label
            ]
            motion_records.sort(key=lambda record: record["time_sec"])
            index[scene_name][motion_label]["records"] = motion_records
            index[scene_name][motion_label]["times"] = [
                record["time_sec"] for record in motion_records
            ]
            index[scene_name][motion_label]["laps"] = lap_infos
    return index


@lru_cache(maxsize=256)
def _read_rgb_cached(rgb_path):
    return cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)


def _center_patch_bounds(img_shape, patch_size=RGB_MATCH_PATCH_SIZE):
    height, width = img_shape[:2]
    patch_width = min(int(patch_size), int(width))
    patch_height = min(int(patch_size), int(height))
    x0 = max((int(width) - patch_width) // 2, 0)
    y0 = max((int(height) - patch_height) // 2, 0)
    return x0, y0, patch_width, patch_height


def _rgb_patch_difference(source_img, candidate_img):
    if source_img is None or candidate_img is None:
        return None

    if source_img.shape != candidate_img.shape:
        candidate_img = cv2.resize(
            candidate_img,
            (source_img.shape[1], source_img.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    full_diff = source_img.astype(np.float32) - candidate_img.astype(np.float32)
    full_abs_diff = np.abs(full_diff)
    x0, y0, patch_width, patch_height = _center_patch_bounds(source_img.shape)
    source_patch = source_img[y0:y0 + patch_height, x0:x0 + patch_width]
    candidate_patch = candidate_img[y0:y0 + patch_height, x0:x0 + patch_width]
    diff = source_patch.astype(np.float32) - candidate_patch.astype(np.float32)
    abs_diff = np.abs(diff)
    patch_mean_abs_diff = float(abs_diff.mean())
    patch_rmse_diff = float(np.sqrt(np.mean(diff ** 2)))
    patch_max_abs_diff = float(abs_diff.max())
    return {
        "rgb_patch_x0": int(x0),
        "rgb_patch_y0": int(y0),
        "rgb_patch_width": int(patch_width),
        "rgb_patch_height": int(patch_height),
        "rgb_patch_mean_abs_diff": patch_mean_abs_diff,
        "rgb_patch_rmse_diff": patch_rmse_diff,
        "rgb_patch_max_abs_diff": patch_max_abs_diff,
        "rgb_mean_abs_diff": float(full_abs_diff.mean()),
        "rgb_rmse_diff": float(np.sqrt(np.mean(full_diff ** 2))),
        "rgb_max_abs_diff": float(full_abs_diff.max()),
    }


@lru_cache(maxsize=8192)
def _load_depth_meters(depth_path):
    depth = np.load(depth_path).astype(np.float32) / 1000
    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid) and float(np.nanmedian(depth[valid])) > DEPTH_AUTO_SCALE_THRESHOLD:
        depth = depth * DEPTH_RAW_TO_METER_SCALE
    return depth


@lru_cache(maxsize=8192)
def _depth_frame_feature(depth_path):
    depth = _load_depth_meters(depth_path)
    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros(DTW_DEPTH_FEATURE_SIZE[0] * DTW_DEPTH_FEATURE_SIZE[1], dtype=np.float32)

    fill_value = float(np.nanmedian(depth[valid]))
    depth = np.where(valid, depth, fill_value).astype(np.float32)
    small = cv2.resize(
        depth,
        (DTW_DEPTH_FEATURE_SIZE[1], DTW_DEPTH_FEATURE_SIZE[0]),
        interpolation=cv2.INTER_AREA,
    )
    return small.reshape(-1).astype(np.float32)


lap_depth_feature_cache = {}


def _lap_cache_key(records):
    if not records:
        return None
    first = records[0]
    return (first["scene"], first["pair_dir"], first["lap_dir_name"])


def _sample_records_for_dtw(records, max_len=DTW_MAX_SEQUENCE_LENGTH):
    if len(records) <= max_len:
        return records
    indices = np.linspace(0, len(records) - 1, max_len).round().astype(int)
    return [records[int(idx)] for idx in indices]


def _lap_depth_feature_sequence(records):
    cache_key = _lap_cache_key(records)
    if cache_key in lap_depth_feature_cache:
        return lap_depth_feature_cache[cache_key]

    sampled_records = _sample_records_for_dtw(records)
    features = []
    for record in sampled_records:
        depth_path = record["depth_path"]
        if Path(depth_path).exists():
            features.append(_depth_frame_feature(depth_path))

    if features:
        sequence = np.stack(features, axis=0).astype(np.float32)
    else:
        sequence = np.empty((0, DTW_DEPTH_FEATURE_SIZE[0] * DTW_DEPTH_FEATURE_SIZE[1]), dtype=np.float32)

    if cache_key is not None:
        lap_depth_feature_cache[cache_key] = sequence
    return sequence


def _dtw_distance(seq_a, seq_b):
    if len(seq_a) == 0 or len(seq_b) == 0:
        return float("inf")

    cost = np.mean(np.abs(seq_a[:, None, :] - seq_b[None, :, :]), axis=2)
    n, m = cost.shape
    prev = np.full(m + 1, np.inf, dtype=np.float32)
    curr = np.full(m + 1, np.inf, dtype=np.float32)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = np.inf
        for j in range(1, m + 1):
            curr[j] = cost[i - 1, j - 1] + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / max(n + m, 1))


def _select_best_canonical_lap(source_lap_records, candidate_laps):
    source_seq = _lap_depth_feature_sequence(source_lap_records)
    best_lap = None
    best_cost = float("inf")
    for lap_info in candidate_laps:
        candidate_seq = _lap_depth_feature_sequence(lap_info["records"])
        dtw_cost = _dtw_distance(source_seq, candidate_seq)
        if dtw_cost < best_cost:
            best_cost = dtw_cost
            best_lap = lap_info
    return best_lap, best_cost


def _nearest_time_record(records, times, source_time_sec):
    if not records:
        return None
    insert_pos = bisect_left(times, source_time_sec)
    candidate_indices = []
    if insert_pos < len(records):
        candidate_indices.append(insert_pos)
    if insert_pos > 0:
        candidate_indices.append(insert_pos - 1)
    best_idx = min(candidate_indices, key=lambda idx: abs(times[idx] - source_time_sec))
    return records[best_idx]


def _depth_mean_abs_difference(source_depth_path, candidate_depth_path):
    source_depth = _load_depth_meters(source_depth_path)
    candidate_depth = _load_depth_meters(candidate_depth_path)
    if source_depth.shape != candidate_depth.shape:
        candidate_depth = cv2.resize(
            candidate_depth,
            (source_depth.shape[1], source_depth.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    valid = np.isfinite(source_depth) & np.isfinite(candidate_depth)
    if valid.sum() < 10:
        return None
    return float(np.mean(np.abs(source_depth[valid] - candidate_depth[valid])))


def _registration_default_metrics(status):
    return {
        "registered_depth_mean_abs_diff": NO_MATCH_VALUE,
        "registered_depth_rmse_diff": NO_MATCH_VALUE,
        "registered_depth_max_abs_diff": NO_MATCH_VALUE,
        "registration_overlap_ratio": NO_MATCH_VALUE,
        "registration_ecc_score": NO_MATCH_VALUE,
        "registration_dx_px": NO_MATCH_VALUE,
        "registration_dy_px": NO_MATCH_VALUE,
        "registration_status": status,
    }


def _resize_affine_to_full_resolution(warp_matrix, small_shape, full_shape):
    small_h, small_w = small_shape[:2]
    full_h, full_w = full_shape[:2]
    sx = small_w / max(full_w, 1)
    sy = small_h / max(full_h, 1)
    scale_to_small = np.array(
        [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    scale_to_full = np.array(
        [[1.0 / sx, 0.0, 0.0], [0.0, 1.0 / sy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    warp_h = np.eye(3, dtype=np.float32)
    warp_h[:2, :] = warp_matrix
    full_warp = scale_to_full @ warp_h @ scale_to_small
    return full_warp[:2, :].astype(np.float32)


def _depth_registration_image(depth, valid_mask):
    width, height = REGISTRATION_IMAGE_SIZE
    depth_small = cv2.resize(depth, (width, height), interpolation=cv2.INTER_AREA)
    valid_small = cv2.resize(
        valid_mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    finite = np.isfinite(depth_small) & valid_small
    if finite.sum() < 10:
        return None, None

    values = depth_small[finite]
    lo, hi = np.percentile(values, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, None

    image = np.zeros_like(depth_small, dtype=np.float32)
    image[finite] = np.clip((depth_small[finite] - lo) / (hi - lo), 0.0, 1.0)
    return image.astype(np.float32), finite


def _registered_depth_difference(source_depth_path, candidate_depth_path):
    source_depth = _load_depth_meters(source_depth_path)
    candidate_depth = _load_depth_meters(candidate_depth_path)
    if source_depth.shape != candidate_depth.shape:
        candidate_depth = cv2.resize(
            candidate_depth,
            (source_depth.shape[1], source_depth.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    source_valid = np.isfinite(source_depth) & (source_depth > MIN_DEPTH) & (source_depth < MAX_DEPTH)
    candidate_valid = np.isfinite(candidate_depth) & (candidate_depth > MIN_DEPTH) & (candidate_depth < MAX_DEPTH)
    if source_valid.sum() < 10 or candidate_valid.sum() < 10:
        return None

    source_reg, source_reg_valid = _depth_registration_image(source_depth, source_valid)
    candidate_reg, _candidate_reg_valid = _depth_registration_image(candidate_depth, candidate_valid)
    if source_reg is None or candidate_reg is None:
        return None

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        REGISTRATION_ECC_MAX_ITERS,
        REGISTRATION_ECC_EPS,
    )
    try:
        ecc_score, warp = cv2.findTransformECC(
            source_reg,
            candidate_reg,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            source_reg_valid.astype(np.uint8),
            5,
        )
    except cv2.error:
        return None

    full_warp = _resize_affine_to_full_resolution(warp, source_reg.shape, source_depth.shape)
    output_size = (source_depth.shape[1], source_depth.shape[0])
    aligned_candidate = cv2.warpAffine(
        candidate_depth,
        full_warp,
        output_size,
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    aligned_valid = cv2.warpAffine(
        candidate_valid.astype(np.uint8),
        full_warp,
        output_size,
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    valid = source_valid & aligned_valid & np.isfinite(aligned_candidate)
    overlap_ratio = float(valid.sum() / max(source_valid.sum(), 1))
    if valid.sum() < 10:
        return None

    diff = source_depth[valid].astype(np.float32) - aligned_candidate[valid].astype(np.float32)
    abs_diff = np.abs(diff)
    return {
        "registered_depth_mean_abs_diff": float(abs_diff.mean()),
        "registered_depth_rmse_diff": float(np.sqrt(np.mean(diff ** 2))),
        "registered_depth_max_abs_diff": float(abs_diff.max()),
        "registration_overlap_ratio": overlap_ratio,
        "registration_ecc_score": float(ecc_score),
        "registration_dx_px": float(full_warp[0, 2]),
        "registration_dy_px": float(full_warp[1, 2]),
        "registration_status": "registered",
    }


def _nearest_time_candidate_indices(times, source_time_sec):
    if not times:
        return []

    insert_pos = bisect_left(times, source_time_sec)
    left = max(0, insert_pos - CANONICAL_TOPK_TIME_CANDIDATES)
    right = min(len(times), insert_pos + CANONICAL_TOPK_TIME_CANDIDATES + 1)
    return sorted(
        range(left, right),
        key=lambda idx: abs(times[idx] - source_time_sec),
    )[:CANONICAL_TOPK_TIME_CANDIDATES]


def _nearest_time_candidate_records(records, times, source_time_sec):
    return [
        records[idx]
        for idx in _nearest_time_candidate_indices(times, source_time_sec)
    ]


def _find_best_canonical_match(source_record, source_lap_records, canonical_entry):
    canonical_lap, dtw_cost = _select_best_canonical_lap(
        source_lap_records,
        canonical_entry.get("laps", []),
    )
    if canonical_lap is None:
        return None

    motion_label = source_record["motion_label"]
    candidate_records = canonical_lap["motion_records"].get(motion_label, [])
    candidate_times = canonical_lap["motion_times"].get(motion_label, [])
    candidate_records = _nearest_time_candidate_records(
        candidate_records,
        candidate_times,
        source_record["time_sec"],
    )
    if not candidate_records:
        return None

    best_match = None
    best_score = float("inf")
    for candidate_record in candidate_records:
        time_diff_sec = abs(source_record["time_sec"] - candidate_record["time_sec"])
        raw_depth_mean_abs_diff = _depth_mean_abs_difference(
            source_record["depth_path"],
            candidate_record["depth_path"],
        )
        registration_metrics = _registered_depth_difference(
            source_record["depth_path"],
            candidate_record["depth_path"],
        )
        if registration_metrics is None:
            continue

        registration_overlap_ratio = registration_metrics["registration_overlap_ratio"]
        registered_depth_mean_abs_diff = registration_metrics["registered_depth_mean_abs_diff"]
        registered_ecc_scores = registration_metrics["registration_ecc_score"]
        if registration_overlap_ratio < REGISTRATION_MIN_OVERLAP_RATIO:
            registration_status = "registration_overlap_too_low"
        elif registered_depth_mean_abs_diff > DEPTH_MATCH_MAX_MEAN_ABS_DIFF:
            registration_status = "registered_depth_diff_too_large"
        elif registered_ecc_scores < DEPTH_MATCH_MIN_ECC_SCORE:
            registration_status = "registered_ecc_score_is_too_low"
        else:
            registration_status = "registered"

        score = (
            registered_depth_mean_abs_diff
            + REGISTRATION_SCORE_TIME_WEIGHT * float(time_diff_sec)
        )
        candidate_match = {
            "record": candidate_record,
            "time_diff_sec": float(time_diff_sec),
            "matched_lap_dtw_cost": float(dtw_cost),
            "raw_depth_mean_abs_diff": (
                NO_MATCH_VALUE if raw_depth_mean_abs_diff is None else float(raw_depth_mean_abs_diff)
            ),
            "depth_mean_abs_diff": float(registered_depth_mean_abs_diff),
            "depth_rejected": registration_status != "registered",
            **registration_metrics,
            "registration_status": registration_status,
        }
        if score < best_score:
            best_score = score
            best_match = candidate_match

    if best_match is None:
        return {
            "registration_failed": True,
            "matched_lap_dtw_cost": float(dtw_cost),
        }

    source_img = cv2.imread(source_record["rgb_path"], cv2.IMREAD_COLOR)
    candidate_img = _read_rgb_cached(best_match["record"]["rgb_path"])
    rgb_metrics = _rgb_patch_difference(source_img, candidate_img)
    if rgb_metrics is None:
        rgb_metrics = {
            "rgb_patch_x0": NO_MATCH_VALUE,
            "rgb_patch_y0": NO_MATCH_VALUE,
            "rgb_patch_width": NO_MATCH_VALUE,
            "rgb_patch_height": NO_MATCH_VALUE,
            "rgb_patch_mean_abs_diff": NO_MATCH_VALUE,
            "rgb_patch_rmse_diff": NO_MATCH_VALUE,
            "rgb_patch_max_abs_diff": NO_MATCH_VALUE,
            "rgb_mean_abs_diff": NO_MATCH_VALUE,
            "rgb_rmse_diff": NO_MATCH_VALUE,
            "rgb_max_abs_diff": NO_MATCH_VALUE,
        }

    best_match.update(rgb_metrics)
    return best_match


def _source_row_base(source_record):
    return {
        "scene": source_record["scene"],
        "source_pair_index": source_record["pair_index"],
        "source_pair_dir": source_record["pair_dir"],
        "source_exposure": source_record["exposure"],
        "source_gain": source_record["gain"],
        "source_lap_dir": source_record["lap_dir_name"],
        "source_frame_index": source_record["frame_index"],
        "source_time_sec": source_record["time_sec"],
        "source_motion_label": source_record["motion_label"],
        "source_rgb_path": source_record["rgb_path"],
        "source_depth_path": source_record["depth_path"],
    }


def _no_match_fields(match_status, canonical_param=None):
    row = {
        "canonical_exposure": NO_MATCH_VALUE,
        "canonical_gain": NO_MATCH_VALUE,
        "canonical_pair_index": NO_MATCH_VALUE,
        "canonical_pair_dir": NO_MATCH_VALUE,
        "matched_lap_dir": NO_MATCH_VALUE,
        "matched_frame_index": NO_MATCH_VALUE,
        "matched_time_sec": NO_MATCH_VALUE,
        "time_diff_sec": NO_MATCH_VALUE,
        "matched_lap_dtw_cost": NO_MATCH_VALUE,
        "raw_depth_mean_abs_diff": NO_MATCH_VALUE,
        "depth_mean_abs_diff": NO_MATCH_VALUE,
        "depth_diff_threshold": DEPTH_MATCH_MAX_MEAN_ABS_DIFF,
        "rgb_patch_x0": NO_MATCH_VALUE,
        "rgb_patch_y0": NO_MATCH_VALUE,
        "rgb_patch_width": NO_MATCH_VALUE,
        "rgb_patch_height": NO_MATCH_VALUE,
        "rgb_patch_mean_abs_diff": NO_MATCH_VALUE,
        "rgb_patch_rmse_diff": NO_MATCH_VALUE,
        "rgb_patch_max_abs_diff": NO_MATCH_VALUE,
        "rgb_mean_abs_diff": NO_MATCH_VALUE,
        "rgb_rmse_diff": NO_MATCH_VALUE,
        "rgb_max_abs_diff": NO_MATCH_VALUE,
        "matched_rgb_path": NO_MATCH_VALUE,
        "matched_depth_path": NO_MATCH_VALUE,
        "match_status": match_status,
        **_registration_default_metrics("not_run"),
    }
    if canonical_param is not None:
        exposure, gain, pair_index, pair_dir = canonical_param
        row.update({
            "canonical_exposure": exposure,
            "canonical_gain": gain,
            "canonical_pair_index": pair_index,
            "canonical_pair_dir": pair_dir,
        })
    return row


def _match_fields(best_match, canonical_param):
    exposure, gain, pair_index, pair_dir = canonical_param
    matched_record = best_match["record"]
    return {
        "canonical_exposure": exposure,
        "canonical_gain": gain,
        "canonical_pair_index": pair_index,
        "canonical_pair_dir": pair_dir,
        "matched_lap_dir": matched_record["lap_dir_name"],
        "matched_frame_index": matched_record["frame_index"],
        "matched_time_sec": matched_record["time_sec"],
        "time_diff_sec": best_match["time_diff_sec"],
        "matched_lap_dtw_cost": best_match["matched_lap_dtw_cost"],
        "raw_depth_mean_abs_diff": best_match["raw_depth_mean_abs_diff"],
        "depth_mean_abs_diff": best_match["depth_mean_abs_diff"],
        "depth_diff_threshold": DEPTH_MATCH_MAX_MEAN_ABS_DIFF,
        "registered_depth_mean_abs_diff": best_match["registered_depth_mean_abs_diff"],
        "registered_depth_rmse_diff": best_match["registered_depth_rmse_diff"],
        "registered_depth_max_abs_diff": best_match["registered_depth_max_abs_diff"],
        "registration_overlap_ratio": best_match["registration_overlap_ratio"],
        "registration_ecc_score": best_match["registration_ecc_score"],
        "registration_dx_px": best_match["registration_dx_px"],
        "registration_dy_px": best_match["registration_dy_px"],
        "registration_status": best_match["registration_status"],
        "rgb_patch_x0": best_match["rgb_patch_x0"],
        "rgb_patch_y0": best_match["rgb_patch_y0"],
        "rgb_patch_width": best_match["rgb_patch_width"],
        "rgb_patch_height": best_match["rgb_patch_height"],
        "rgb_patch_mean_abs_diff": best_match["rgb_patch_mean_abs_diff"],
        "rgb_patch_rmse_diff": best_match["rgb_patch_rmse_diff"],
        "rgb_patch_max_abs_diff": best_match["rgb_patch_max_abs_diff"],
        "rgb_mean_abs_diff": best_match["rgb_mean_abs_diff"],
        "rgb_rmse_diff": best_match["rgb_rmse_diff"],
        "rgb_max_abs_diff": best_match["rgb_max_abs_diff"],
        "matched_rgb_path": matched_record["rgb_path"],
        "matched_depth_path": matched_record["depth_path"],
        "match_status": "matched",
    }


def _match_row_for_source_record(source_record, source_lap_records, canonical_frame_index):
    row = _source_row_base(source_record)
    motion_label = source_record["motion_label"]
    canonical_param = _canonical_param_tuple(
        canonical_parameter.get(source_record["scene"], {}).get(motion_label)
    )

    if canonical_param is None:
        row.update(_no_match_fields("no_canonical_parameter"))
        return row

    canonical_entry = canonical_frame_index[source_record["scene"]][motion_label]
    if not canonical_entry["records"]:
        row.update(_no_match_fields("no_canonical_rgb_candidates", canonical_param))
        return row

    best_match = _find_best_canonical_match(source_record, source_lap_records, canonical_entry)
    if best_match is None:
        row.update(_no_match_fields("no_canonical_frame_after_dtw", canonical_param))
        return row

    if best_match.get("registration_failed"):
        row.update(_no_match_fields("depth_registration_failed", canonical_param))
        row.update({
            "matched_lap_dtw_cost": best_match["matched_lap_dtw_cost"],
            "registration_status": "failed",
        })
        return row

    if best_match.get("depth_rejected"):
        reject_status = best_match.get("registration_status") or "depth_diff_too_large"
        row.update(_no_match_fields(reject_status, canonical_param))
        row.update({
            "matched_lap_dtw_cost": best_match["matched_lap_dtw_cost"],
            "raw_depth_mean_abs_diff": best_match["raw_depth_mean_abs_diff"],
            "depth_mean_abs_diff": best_match["depth_mean_abs_diff"],
            "depth_diff_threshold": DEPTH_MATCH_MAX_MEAN_ABS_DIFF,
            "registered_depth_mean_abs_diff": best_match["registered_depth_mean_abs_diff"],
            "registered_depth_rmse_diff": best_match["registered_depth_rmse_diff"],
            "registered_depth_max_abs_diff": best_match["registered_depth_max_abs_diff"],
            "registration_overlap_ratio": best_match["registration_overlap_ratio"],
            "registration_ecc_score": best_match["registration_ecc_score"],
            "registration_dx_px": best_match["registration_dx_px"],
            "registration_dy_px": best_match["registration_dy_px"],
            "registration_status": best_match["registration_status"],
            "rgb_patch_x0": best_match["rgb_patch_x0"],
            "rgb_patch_y0": best_match["rgb_patch_y0"],
            "rgb_patch_width": best_match["rgb_patch_width"],
            "rgb_patch_height": best_match["rgb_patch_height"],
            "rgb_patch_mean_abs_diff": best_match["rgb_patch_mean_abs_diff"],
            "rgb_patch_rmse_diff": best_match["rgb_patch_rmse_diff"],
            "rgb_patch_max_abs_diff": best_match["rgb_patch_max_abs_diff"],
            "rgb_mean_abs_diff": best_match["rgb_mean_abs_diff"],
            "rgb_rmse_diff": best_match["rgb_rmse_diff"],
            "rgb_max_abs_diff": best_match["rgb_max_abs_diff"],
        })
        return row

    row.update(_match_fields(best_match, canonical_param))
    return row


def _reset_canonical_match_outputs():
    CANONICAL_MATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANONICAL_MATCH_OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def _write_canonical_matches_for_scene(scene_name, canonical_frame_index=None, append_global=True):
    scene_path = Path(DATA_PATH) / scene_name
    if not scene_path.exists():
        print(f"Skip missing scene: {scene_path}")
        return {"scene": scene_name, "rows_written": 0, "rows_without_match": 0, "output_csv": None}

    if canonical_frame_index is None:
        canonical_frame_index = _build_canonical_frame_index(scene_name)

    CANONICAL_MATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_output_csv = CANONICAL_MATCH_OUTPUT_DIR / f"{scene_name}_canonical_frame_matches.csv"
    global_needs_header = not CANONICAL_MATCH_OUTPUT_CSV.exists() or CANONICAL_MATCH_OUTPUT_CSV.stat().st_size == 0
    rows_written = 0
    rows_without_match = 0

    with open(scene_output_csv, "w", newline="") as scene_csv_file:
        scene_writer = csv.DictWriter(scene_csv_file, fieldnames=CSV_COLUMNS)
        scene_writer.writeheader()

        global_csv_file = open(CANONICAL_MATCH_OUTPUT_CSV, "a", newline="") if append_global else None
        try:
            global_writer = None
            if global_csv_file is not None:
                global_writer = csv.DictWriter(global_csv_file, fieldnames=CSV_COLUMNS)
                if global_needs_header:
                    global_writer.writeheader()

            for pair_index, (exposure, gain, p_prefix) in enumerate(PARAM_PAIRS):
                if exposure == 2000: continue
                pair_dir = f"pair_{pair_index:03d}_{p_prefix}"
                pair_path = scene_path / pair_dir
                if not pair_path.exists():
                    continue

                for lap_dir in sorted(pair_path.glob("lap_*")):
                    source_lap_records = list(_iter_frame_records(
                        scene_name, pair_index, exposure, gain, pair_dir, lap_dir
                    ))
                    source_lap_records = [
                        record for record in source_lap_records
                        if Path(record["rgb_path"]).exists() and Path(record["depth_path"]).exists()
                    ]
                    source_lap_records.sort(key=lambda record: record["time_sec"])
                    for source_record in source_lap_records:
                        row = _match_row_for_source_record(
                            source_record,
                            source_lap_records,
                            canonical_frame_index,
                        )
                        if row["match_status"] != "matched":
                            rows_without_match += 1

                        scene_writer.writerow(row)
                        if global_writer is not None:
                            global_writer.writerow(row)
                        rows_written += 1
        finally:
            if global_csv_file is not None:
                global_csv_file.close()

    print(
        f"Saved {rows_written} rows to {scene_output_csv} "
        f"({rows_without_match} rows with -1 match fields)."
    )
    return {
        "scene": scene_name,
        "rows_written": rows_written,
        "rows_without_match": rows_without_match,
        "output_csv": scene_output_csv,
    }

# %% Cell 4
from tqdm import tqdm

DATA_PATH = os.environ.get("DATA_PATH", "/home/kimh060612/ati_workspace/orbbec_realworld_dataset") # orbbec_realworld_dataset" # realworld_dataset"

SCENE_PREFIX = "comlab_scene"
EXPOSURE_SET = [2000, 4000, 8000, 16000, 32000] # [20, 40, 80, 160, 320] # [2000, 4000, 8000, 16000, 32000] # 
GAIN_SET = [16, 32, 64, 128]
PARAM_PAIRS = [
    (e, g, f"exposure_{e}_gain_{g}")
    for g in GAIN_SET
    for e in EXPOSURE_SET
]
LIGHT_PREFIX = ["normal", "dim", "dark"]
SPEED_PREFIX = ["fast"] # , "normal"
TARGET_TOPOLOGY = ["topology1", "topology4"]
MOTION_SET = ["stop", "spin", "rotate", "slow", "fast"]

metric_list = {
    f"{SCENE_PREFIX}_{light}_{speed}_{topology}": {
        f"pair_{i:03d}_{p_prefix}": {
            m: [] ### List element: Dict[str, float]
            for m in MOTION_SET
        } 
        for i, (e, g, p_prefix) in enumerate(PARAM_PAIRS) if not e == 2000
    }
    for light, speed, topology in product(LIGHT_PREFIX, SPEED_PREFIX, TARGET_TOPOLOGY)    
}

best_performance = {
    f"{SCENE_PREFIX}_{light}_{speed}_{topology}": {
        m: [987654321, -1.0] ## abs_rel, \delta_{1.25}
        for m in MOTION_SET
    }
    for light, speed, topology in product(LIGHT_PREFIX, SPEED_PREFIX, TARGET_TOPOLOGY)
}

canonical_parameter = {
    f"{SCENE_PREFIX}_{light}_{speed}_{topology}": {
        m: [0.0, 0.0] ## exposure time, gain
        for m in MOTION_SET
    }
    for light, speed, topology in product(LIGHT_PREFIX, SPEED_PREFIX, TARGET_TOPOLOGY)    
}

canonical_match_summaries = []
_reset_canonical_match_outputs()

device = os.environ.get("DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model, processor = load_depth_anything_metric_indoor(device)
for light, speed, topology in product(LIGHT_PREFIX, SPEED_PREFIX, TARGET_TOPOLOGY):
    target_path = f"{SCENE_PREFIX}_{light}_{speed}_{topology}"
    if os.path.exists(f"./{str(CANONICAL_MATCH_OUTPUT_DIR)}/{target_path}_canonical_frame_matches.csv"): 
        print(f"Already {target_path} exists, skipping....")
        continue
    num_laps = 0
    for i, (e, g, p_prefix) in tqdm(enumerate(PARAM_PAIRS)):
        if e == 2000: continue
        subdir_name = f"pair_{i:03d}_{p_prefix}"
        lap_dirs = glob(f"{DATA_PATH}/{target_path}/{subdir_name}/lap_*")
        ## RGB/Depth/metadata를 받아온다. 
        for lap_dir in lap_dirs:
            list_rgb_path = glob(f"{lap_dir}/rgb/*.png")
            f_sec, e_sec = lap_duration_sec(lap_dir)
            N = len(list_rgb_path)
            ## 모든 데이터의 성능을 batch 단위로 측정한다. 
            for batch_start in range(0, N, EVAL_BATCH_SIZE):
                batch_indices = range(batch_start, min(batch_start + EVAL_BATCH_SIZE, N))
                batch_records = []
                rgb_bgr_list = []
                depth_imgs = []
                target_shapes = []

                for idx in batch_indices:
                    rgb_path = f"{lap_dir}/rgb/{idx:06d}.png"
                    depth_path = f"{lap_dir}/depth/{idx:06d}.npy"
                    metadata_path = f"{lap_dir}/metadata/{idx:06d}.json"
                    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                    if rgb_bgr is None:
                        continue
                    if not os.path.exists(depth_path) or not os.path.exists(metadata_path): 
                        continue
                    depth_img = np.load(depth_path).astype(np.float32) / 1000
                    metadata_json = json.load(open(metadata_path, "r"))
                    
                    curr_time = frame_time_sec(metadata_json) - f_sec
                    accel, linear_vel, yaw_ang = get_motion_data(metadata_json)
                    motion_label = assign_motion_label(linear_vel, accel, yaw_ang)

                    batch_records.append({
                        "curr_time": curr_time,
                        "rgb_path": rgb_path,
                        "motion_label": motion_label,
                    })
                    rgb_bgr_list.append(rgb_bgr)
                    depth_imgs.append(depth_img)
                    target_shapes.append(tuple(depth_img.shape[:2]))

                if not batch_records:
                    continue

                pred_depths = predict_metric_depth_batch(model, processor, rgb_bgr_list, target_shapes, device)
                pred_tensor = torch.stack(pred_depths, dim=0)
                gt_tensor = torch.from_numpy(np.stack(depth_imgs, axis=0)).to(device=device, dtype=torch.float32)

                valid_mask = (
                    torch.isfinite(ensure_bchw(gt_tensor))
                    & torch.isfinite(ensure_bchw(pred_tensor))
                    & (ensure_bchw(gt_tensor) > MIN_DEPTH)
                    & (ensure_bchw(gt_tensor) < MAX_DEPTH)
                    & (ensure_bchw(pred_tensor) > 0)
                )
                aligned_pred = align_relative_prediction_to_depth_space(
                    pred_tensor,
                    gt_tensor,
                    valid_mask,
                    align_mode="scale_shift",
                )["depth"]
                metric_tensors = compute_comprehensive_depth_metrics(aligned_pred, gt_tensor, valid_mask)
                valid_counts = ensure_bchw(valid_mask).flatten(1).sum(dim=1).detach().cpu().numpy()

                for batch_pos, record in enumerate(batch_records):
                    metric_vals = {
                        key: float(value[batch_pos].detach().cpu())
                        for key, value in metric_tensors.items()
                    }
                    if valid_counts[batch_pos] < 10 or not all(np.isfinite(v) for v in metric_vals.values()):
                        metric_vals = {key: -1.0 for key in metric_tensors.keys()}

                    metric_list[target_path][subdir_name][record["motion_label"]].append({
                        "curr_time": record["curr_time"], 
                        "rgb_path": record["rgb_path"], 
                        **metric_vals
                    })
        num_laps += len(lap_dirs)
    ## 해당 scene에서 가장 성능이 높은 canonical parameter를 결정한다.
    for m in MOTION_SET:
        for i, (e, g, p_prefix) in enumerate(PARAM_PAIRS):
            if e == 2000: continue
            subdir_name = f"pair_{i:03d}_{p_prefix}"        
            metrics = metric_list[target_path][subdir_name][m]
            np_abs_rel = np.array([ metric["abs_rel"] for metric in metrics ])
            np_a1 = np.array([ metric["a1"] for metric in metrics ])
            mean_abs_rel = np.mean(np_abs_rel)
            mean_a1 = np.mean(np_a1)
            if best_performance[target_path][m][0] > mean_abs_rel and best_performance[target_path][m][1] < mean_a1:
                best_performance[target_path][m][0] = mean_abs_rel
                best_performance[target_path][m][1] = mean_a1
                canonical_parameter[target_path][m][0] = e
                canonical_parameter[target_path][m][1] = g    
    ### 해당 scene의 모든 timestep에서 가장 canonical parameter 기준으로 비슷한 RGB image를 selection한다.    
    #### Things to save:
    ##### Most closest timestep, RGB Image, Depth Image를 기록 후, csv로 작성.
    scene_canonical_frame_index = _build_canonical_frame_index(target_path)
    canonical_match_summaries.append(
        _write_canonical_matches_for_scene(
            target_path,
            canonical_frame_index=scene_canonical_frame_index,
            append_global=True,
        )
    )

print("Canonical Parameter per Scene/Motion")
print(canonical_parameter)

# %% Cell 5
### Implement HERE

print("Canonical frame matching is now saved per scene inside the main scene loop.")
print(f"Scene CSV directory: {CANONICAL_MATCH_OUTPUT_DIR}")
print(f"Global appended CSV: {CANONICAL_MATCH_OUTPUT_CSV}")

# %% Cell 7
### Implement Video Save HERE

VIDEO_SCENE_NAME = "comlab_scene_normal_fast_topology1"
VIDEO_MATCH_CSV = CANONICAL_MATCH_OUTPUT_DIR / f"{VIDEO_SCENE_NAME}_canonical_frame_matches.csv"
VIDEO_OUTPUT_DIR = Path("analysis_videos/canonical_match_videos")
VIDEO_OUTPUT_PATH = VIDEO_OUTPUT_DIR / f"{VIDEO_SCENE_NAME}_source_vs_canonical.mp4"
VIDEO_FPS = 15
VIDEO_FRAME_STRIDE = 1
VIDEO_MAX_FRAMES = None
VIDEO_PANEL_WIDTH = 640
VIDEO_HEADER_HEIGHT = 70


def _safe_int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_row_sort_key(row):
    return (
        _safe_int(row.get("source_pair_index")),
        _safe_int(str(row.get("source_lap_dir", "")).replace("lap_", "")),
        _safe_int(row.get("source_frame_index")),
    )


def _load_match_rows(csv_path, frame_stride=1, max_frames=None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Match CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    rows = [row for row in rows if row.get("scene") == VIDEO_SCENE_NAME]
    rows.sort(key=_source_row_sort_key)
    if frame_stride and frame_stride > 1:
        rows = rows[::frame_stride]
    if max_frames is not None:
        rows = rows[:max_frames]
    if not rows:
        raise ValueError(f"No rows to render for scene {VIDEO_SCENE_NAME} in {csv_path}")
    return rows


def _resize_panel(img_bgr, panel_width=VIDEO_PANEL_WIDTH):
    if img_bgr is None:
        img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    height, width = img_bgr.shape[:2]
    scale = panel_width / max(width, 1)
    panel_height = max(int(round(height * scale)), 1)
    return cv2.resize(img_bgr, (panel_width, panel_height), interpolation=cv2.INTER_AREA)


def _draw_roi_box(panel_bgr, row, original_shape, color=(0, 255, 255)):
    x0 = _safe_int(row.get("rgb_patch_x0"))
    y0 = _safe_int(row.get("rgb_patch_y0"))
    patch_w = _safe_int(row.get("rgb_patch_width"))
    patch_h = _safe_int(row.get("rgb_patch_height"))
    if min(x0, y0, patch_w, patch_h) < 0:
        return panel_bgr

    orig_h, orig_w = original_shape[:2]
    scale_x = panel_bgr.shape[1] / max(orig_w, 1)
    scale_y = panel_bgr.shape[0] / max(orig_h, 1)
    p0 = (int(round(x0 * scale_x)), int(round(y0 * scale_y)))
    p1 = (int(round((x0 + patch_w) * scale_x)), int(round((y0 + patch_h) * scale_y)))
    cv2.rectangle(panel_bgr, p0, p1, color, 2, cv2.LINE_AA)
    return panel_bgr


def _add_header(panel_bgr, title, row, is_source):
    header = np.zeros((VIDEO_HEADER_HEIGHT, panel_bgr.shape[1], 3), dtype=np.uint8)
    if is_source:
        line1 = f"SOURCE | {row.get('source_pair_dir')} | {row.get('source_lap_dir')} frame {row.get('source_frame_index')}"
        line2 = f"motion={row.get('source_motion_label')} exp={row.get('source_exposure')} gain={row.get('source_gain')}"
    else:
        line1 = f"MATCHED CANONICAL | {row.get('canonical_pair_dir')} | {row.get('matched_lap_dir')} frame {row.get('matched_frame_index')}"
        line2 = (
            f"exp={row.get('canonical_exposure')} gain={row.get('canonical_gain')} "
            f"dt={_safe_float(row.get('time_diff_sec')):.3f}s "
            f"dtw={_safe_float(row.get('matched_lap_dtw_cost')):.2f} "
            f"regMAD={_safe_float(row.get('registered_depth_mean_abs_diff')):.2f} "
            f"ov={_safe_float(row.get('registration_overlap_ratio')):.2f} "
            f"rgbMAD={_safe_float(row.get('rgb_mean_abs_diff')):.2f} "
            f"status={row.get('match_status')}"
        )
    cv2.putText(header, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, line1[:95], (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(header, line2[:95], (12, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([header, panel_bgr])


def _placeholder_panel(shape, message):
    h, w = shape[:2]
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(panel, message, (20, max(h // 2, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
    return panel


def _make_video_frame(row):
    source_img = cv2.imread(row.get("source_rgb_path", ""), cv2.IMREAD_COLOR)
    if source_img is None:
        source_img = np.zeros((480, 640, 3), dtype=np.uint8)

    matched_path = row.get("matched_rgb_path", "")
    matched_img = None
    if row.get("match_status") == "matched" and matched_path not in (None, "", str(NO_MATCH_VALUE)):
        matched_img = cv2.imread(matched_path, cv2.IMREAD_COLOR)
    if matched_img is None:
        matched_img = _placeholder_panel(source_img.shape, f"NO MATCH: {row.get('match_status')}")

    source_panel = _resize_panel(source_img)
    matched_panel = _resize_panel(matched_img)
    if matched_panel.shape[:2] != source_panel.shape[:2]:
        matched_panel = cv2.resize(
            matched_panel,
            (source_panel.shape[1], source_panel.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    source_panel = _draw_roi_box(source_panel, row, source_img.shape, color=(0, 255, 255))
    matched_panel = _draw_roi_box(matched_panel, row, matched_img.shape, color=(0, 255, 255))
    source_panel = _add_header(source_panel, "Source RGB", row, is_source=True)
    matched_panel = _add_header(matched_panel, "Matched Canonical RGB", row, is_source=False)
    return np.hstack([source_panel, matched_panel])


def make_canonical_match_video(
    scene_name=VIDEO_SCENE_NAME,
    match_csv=VIDEO_MATCH_CSV,
    output_path=VIDEO_OUTPUT_PATH,
    fps=VIDEO_FPS,
    frame_stride=VIDEO_FRAME_STRIDE,
    max_frames=VIDEO_MAX_FRAMES,
):
    global VIDEO_SCENE_NAME
    VIDEO_SCENE_NAME = scene_name
    rows = _load_match_rows(match_csv, frame_stride=frame_stride, max_frames=max_frames)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        for frame_idx, row in enumerate(rows):
            frame = _make_video_frame(row)
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {output_path}")
            writer.write(frame)

            if frame_idx % 250 == 0 or frame_idx + 1 == len(rows):
                print(f"[{frame_idx + 1}/{len(rows)}] {row.get('source_lap_dir')} frame {row.get('source_frame_index')}")
    finally:
        if writer is not None:
            writer.release()

    print(f"Wrote video: {output_path.resolve()}")
    return output_path


canonical_match_video_path = make_canonical_match_video()

