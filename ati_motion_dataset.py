import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


LIGHT_LEVELS = ("dark", "dim", "normal")
MOTION_LEVELS = ("slow", "normal", "fast", "rotate", "spin")
SPEED_LEVELS = MOTION_LEVELS
TRAIN_SCENE_PREFIX = "comlab_scene"
VALIDATION_SCENE_PREFIX = "val_comlab_scene"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
DEPTH_EXTENSIONS = (".npy", ".png", ".tif", ".tiff", ".exr")
METADATA_EXTENSIONS = (".json",)

V_STOP = 0.03
W_STOP = 0.05
A_STOP = 0.35
V_SLOW_MAX = 0.25
V_NORMAL_MAX = 0.85
A_NORMAL = 0.80
A_FAST = 1.50
V_ROTATE_MAX = 0.20
W_ROTATE = 0.25
W_SPIN = 1.05

ATI_PIXEL_VALUES_IDX = 0
ATI_DEPTH_IDX = 1
ATI_VALID_MASK_IDX = 2
ATI_CONDITION_IDX = 3
ATI_CONDITION_STATS_IDX = 4
ATI_STATS_VALID_PIXEL_RATIO_IDX = 0
ATI_STATS_MIN_VALID_DEPTH_RATIO_IDX = 1
ATI_STATS_LIGHT_LABEL_IDX = 2
ATI_STATS_SPEED_LABEL_IDX = 3
ATI_STATS_EXPOSURE_IDX = 4
ATI_STATS_GAIN_IDX = 5

ATISample = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
ATIBatch = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass(frozen=True)
class ATIFrameItem:
    rgb_path: Path
    depth_path: Path
    metadata_path: Path
    scene_name: str
    scene_prefix: str
    light: str
    speed: str
    collection_speed: str
    topology: str
    exposure: float
    gain: float
    lap_id: str
    frame_id: str
    linear_speed: float
    angular_speed: float
    acceleration: float


def _parse_scene_dir_name(
    name: str,
    scene_prefix: str,
    light_levels: Sequence[str],
):
    stem = f"{scene_prefix}_"
    if not name.startswith(stem):
        return None

    parts = name[len(stem):].split("_", maxsplit=2)
    if len(parts) != 3:
        return None

    light, collection_speed, topology = parts
    if light not in light_levels or not topology.startswith("topology"):
        return None

    return scene_prefix, light, collection_speed, topology


def _parse_exposure_dir_name(name: str):
    match = re.fullmatch(
        r"pair_\d+_exposure_([0-9.]+)_gain_([0-9.]+)",
        name,
    )
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def _index_files_by_stem(directory: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    if not directory.is_dir():
        return {}

    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            files[path.stem] = path
    return files


def _finite_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _xy_magnitude(vector) -> Optional[float]:
    if not isinstance(vector, Mapping):
        return None

    x = _finite_float(vector.get("x"))
    y = _finite_float(vector.get("y"))
    if x is None or y is None:
        return None
    return math.hypot(x, y)


def _motion_measurements(metadata: Mapping) -> Optional[Tuple[float, float, float]]:
    wheel_odometry = metadata.get("wheel_odometry")
    imu = metadata.get("imu")
    if not isinstance(wheel_odometry, Mapping) or not isinstance(imu, Mapping):
        return None

    linear_speed = _xy_magnitude(wheel_odometry.get("linear_velocity"))
    acceleration = _xy_magnitude(imu.get("linear_acceleration"))

    angular_velocity = wheel_odometry.get("angular_velocity")
    if not isinstance(angular_velocity, Mapping):
        return None
    yaw = _finite_float(
        angular_velocity.get("yaw_z", angular_velocity.get("z"))
    )

    if linear_speed is None or yaw is None or acceleration is None:
        return None
    return linear_speed, abs(yaw), acceleration


def classify_motion_state(
    linear_speed: float,
    angular_speed: float,
    acceleration: float,
    spin_threshold: float = W_SPIN,
) -> str:
    """Classify one frame as stop/slow/normal/fast/rotate/spin."""
    v = abs(float(linear_speed))
    w = abs(float(angular_speed))
    a = abs(float(acceleration))

    if v < V_STOP and w < W_STOP and a < A_STOP:
        return "stop"

    # Rotation takes priority so low linear velocity does not hide a turn.
    if v < V_ROTATE_MAX and w >= spin_threshold:
        return "spin"
    if v < V_ROTATE_MAX and W_ROTATE <= w < spin_threshold:
        return "rotate"

    if v >= V_NORMAL_MAX:
        return "fast"

    if v >= V_SLOW_MAX:
        return "fast" if a >= A_FAST else "normal"

    # Strong acceleration promotes a low-speed transition by one or two levels.
    if a >= A_FAST:
        return "fast"
    if a >= A_NORMAL:
        return "normal"
    return "slow"


def _log_normalize(value: float, min_value: float, max_value: float) -> float:
    if min_value <= 0.0 or max_value <= 0.0 or math.isclose(min_value, max_value):
        return 0.0

    log_value = math.log(max(value, min_value))
    log_min = math.log(min_value)
    log_max = math.log(max_value)
    return 2.0 * (log_value - log_min) / (log_max - log_min) - 1.0


class _ATIRealWorldDepthMotionBase(Dataset):
    """
    Real-world ATI RGB/Realsense depth dataset.

    Expected layout:
        root/
          {scene_prefix}_{light}_{collection_speed}_{topology}/
            pair_{index}_exposure_{exposure}_gain_{gain}/
              lap_{index}/
                rgb/*.png
                depth/*.npy
                metadata/*.json
                    
    Returned condition vector:
        [light one-hot, motion one-hot, normalized exposure, normalized gain]

    STOP frames and frames without synchronized wheel odometry/IMU are omitted.
    """

    scene_prefix = ""
    split_name = ""

    def __init__(
        self,
        root_dir: str = "/media/michael/ssd1/AIoT_ATI/realworld_dataset",
        image_processor=None,
        image_size: Optional[Tuple[int, int]] = (518, 518),
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        min_valid_depth_ratio: float = 0.3,
        light_levels: Sequence[str] = LIGHT_LEVELS,
        speed_levels: Sequence[str] = SPEED_LEVELS,
        spin_threshold: float = W_SPIN,
        max_samples: Optional[int] = None,
    ):
        if not self.scene_prefix or not self.split_name:
            raise TypeError("Use a concrete training or validation dataset class")

        self.root_dir = Path(root_dir)
        self.image_processor = image_processor
        self.image_size = image_size
        self.split = self.split_name
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_valid_depth_ratio = min_valid_depth_ratio
        self.light_levels = tuple(light_levels)
        self.speed_levels = tuple(speed_levels)
        self.spin_threshold = float(spin_threshold)

        if len(set(self.light_levels)) != len(self.light_levels):
            raise ValueError(f"light_levels contains duplicates: {self.light_levels}")
        if set(self.speed_levels) != set(MOTION_LEVELS):
            raise ValueError(
                "speed_levels must contain exactly "
                f"{MOTION_LEVELS}, got {self.speed_levels}"
            )
        if self.spin_threshold <= W_ROTATE:
            raise ValueError(
                f"spin_threshold must be greater than {W_ROTATE}, "
                f"got {self.spin_threshold}"
            )

        self.light_to_idx = {name: idx for idx, name in enumerate(self.light_levels)}
        self.speed_to_idx = {name: idx for idx, name in enumerate(self.speed_levels)}
        self.condition_names = (
            *[f"light_{name}" for name in self.light_levels],
            *[f"speed_{name}" for name in self.speed_levels],
            "exposure_log_norm",
            "gain_log_norm",
        )

        all_items = self._scan_items()
        if not all_items:
            raise FileNotFoundError(f"No ATI RGB/depth pairs found under {self.root_dir}")

        exposures = [item.exposure for item in all_items]
        gains = [item.gain for item in all_items]
        self.exposure_min = min(exposures)
        self.exposure_max = max(exposures)
        self.gain_min = min(gains)
        self.gain_max = max(gains)

        if max_samples is not None and max_samples > 0:
            all_items = all_items[:max_samples]

        self.items = all_items

    @property
    def condition_dim(self) -> int:
        return len(self.condition_names)

    def _scan_items(self) -> List[ATIFrameItem]:
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        items = []
        scan_stats = {
            "candidate_frames": 0,
            "included_frames": 0,
            "stop_frames": 0,
            "missing_sensor_frames": 0,
            "invalid_metadata_frames": 0,
            "missing_paired_files": 0,
        }
        for scene_dir in sorted(self.root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            parsed_scene = _parse_scene_dir_name(
                scene_dir.name,
                self.scene_prefix,
                self.light_levels,
            )
            if parsed_scene is None:
                continue

            scene_prefix, light, collection_speed, topology = parsed_scene

            for exposure_dir in sorted(scene_dir.iterdir()):
                if not exposure_dir.is_dir():
                    continue

                parsed_exposure = _parse_exposure_dir_name(exposure_dir.name)
                if parsed_exposure is None:
                    continue

                exposure, gain = parsed_exposure
                for lap_dir in sorted(exposure_dir.iterdir()):
                    if (
                        not lap_dir.is_dir()
                        or re.fullmatch(r"lap_\d+", lap_dir.name) is None
                    ):
                        continue

                    rgb_files = _index_files_by_stem(
                        lap_dir / "rgb",
                        IMAGE_EXTENSIONS,
                    )
                    depth_files = _index_files_by_stem(
                        lap_dir / "depth",
                        DEPTH_EXTENSIONS,
                    )
                    metadata_files = _index_files_by_stem(
                        lap_dir / "metadata",
                        METADATA_EXTENSIONS,
                    )
                    paired_frame_ids = set(rgb_files) & set(depth_files)
                    frame_ids = paired_frame_ids & set(metadata_files)
                    scan_stats["missing_paired_files"] += (
                        len(set(rgb_files) | set(depth_files) | set(metadata_files))
                        - len(frame_ids)
                    )

                    for frame_id in sorted(frame_ids):
                        scan_stats["candidate_frames"] += 1
                        metadata_path = metadata_files[frame_id]
                        try:
                            with metadata_path.open("r", encoding="utf-8") as f:
                                metadata = json.load(f)
                        except (OSError, json.JSONDecodeError):
                            scan_stats["invalid_metadata_frames"] += 1
                            continue

                        measurements = _motion_measurements(metadata)
                        if measurements is None:
                            scan_stats["missing_sensor_frames"] += 1
                            continue

                        linear_speed, angular_speed, acceleration = measurements
                        speed = classify_motion_state(
                            linear_speed,
                            angular_speed,
                            acceleration,
                            spin_threshold=self.spin_threshold,
                        )
                        if speed == "stop":
                            scan_stats["stop_frames"] += 1
                            continue

                        items.append(
                            ATIFrameItem(
                                rgb_path=rgb_files[frame_id],
                                depth_path=depth_files[frame_id],
                                metadata_path=metadata_path,
                                scene_name=scene_dir.name,
                                scene_prefix=scene_prefix,
                                light=light,
                                speed=speed,
                                collection_speed=collection_speed,
                                topology=topology,
                                exposure=exposure,
                                gain=gain,
                                lap_id=lap_dir.name,
                                frame_id=frame_id,
                                linear_speed=linear_speed,
                                angular_speed=angular_speed,
                                acceleration=acceleration,
                            )
                        )
                        scan_stats["included_frames"] += 1

        self.scan_stats = scan_stats
        return items

    def __len__(self):
        return len(self.items)

    def _load_depth(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            depth = np.load(path).astype(np.float32)
        else:
            depth = np.array(Image.open(path)).astype(np.float32)
            if np.nanmax(depth) > 100.0:
                depth = depth / 1000.0

        if depth.ndim == 3:
            depth = depth.squeeze()

        return depth.astype(np.float32)

    def _resize_depth_nearest(self, depth: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        height, width = size
        depth_img = Image.fromarray(depth.astype(np.float32))
        depth_img = depth_img.resize((width, height), Image.NEAREST)
        return np.array(depth_img).astype(np.float32)

    def _make_condition(self, item: ATIFrameItem) -> torch.Tensor:
        values = [0.0] * self.condition_dim
        values[self.light_to_idx[item.light]] = 1.0

        speed_offset = len(self.light_levels)
        values[speed_offset + self.speed_to_idx[item.speed]] = 1.0

        values[-2] = _log_normalize(item.exposure, self.exposure_min, self.exposure_max)
        values[-1] = _log_normalize(item.gain, self.gain_min, self.gain_max)

        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, idx: int) -> ATISample:
        item = self.items[idx]

        image = Image.open(item.rgb_path).convert("RGB")
        depth = self._load_depth(item.depth_path)

        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.BICUBIC)
            depth = self._resize_depth_nearest(depth, self.image_size)

        if self.image_processor is None:
            pixel_values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        else:
            inputs = self.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)

        depth = torch.from_numpy(depth).float()
        valid_mask = torch.isfinite(depth)
        valid_mask &= depth > self.min_depth
        valid_mask &= depth < self.max_depth
        valid_pixel_ratio = valid_mask.float().mean()
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            pixel_values,
            depth,
            valid_mask.float(),
            self._make_condition(item),
            torch.tensor(
                [
                    float(valid_pixel_ratio.item()),
                    self.min_valid_depth_ratio,
                    float(self.light_to_idx[item.light]),
                    float(self.speed_to_idx[item.speed]),
                    item.exposure,
                    item.gain,
                ],
                dtype=torch.float32,
            ),
        )


class ATIRealWorldDepthMotionDataset(_ATIRealWorldDepthMotionBase):
    """Training-only dataset backed by comlab_scene_* directories."""

    scene_prefix = TRAIN_SCENE_PREFIX
    split_name = "train"


class ATIRealWorldDepthMotionValidationDataset(_ATIRealWorldDepthMotionBase):
    """Validation-only dataset backed by val_comlab_scene_* directories."""

    scene_prefix = VALIDATION_SCENE_PREFIX
    split_name = "validation"


def ati_collate_fn(batch: List[ATISample]) -> Optional[ATIBatch]:
    batch = [
        sample
        for sample in batch
        if (
            sample[ATI_CONDITION_STATS_IDX][ATI_STATS_VALID_PIXEL_RATIO_IDX].item()
            >= sample[ATI_CONDITION_STATS_IDX][ATI_STATS_MIN_VALID_DEPTH_RATIO_IDX].item()
        )
    ]

    if not batch:
        return None

    return (
        torch.stack([sample[ATI_PIXEL_VALUES_IDX] for sample in batch], dim=0),
        torch.stack([sample[ATI_DEPTH_IDX] for sample in batch], dim=0),
        torch.stack([sample[ATI_VALID_MASK_IDX] for sample in batch], dim=0),
        torch.stack([sample[ATI_CONDITION_IDX] for sample in batch], dim=0),
        torch.stack([sample[ATI_CONDITION_STATS_IDX] for sample in batch], dim=0),
    )


class RGBDepthDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_processor,
        image_size: Optional[Tuple[int, int]] = None,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
    ):
        self.items = []
        self.image_processor = image_processor
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append((row["image_path"], row["depth_path"]))

    def __len__(self):
        return len(self.items)

    def load_depth(self, path: str) -> np.ndarray:
        if path.endswith(".npy"):
            depth = np.load(path).astype(np.float32)
        else:
            depth = np.array(Image.open(path)).astype(np.float32)
            depth = depth / 1000.0
        return depth

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, depth_path = self.items[idx]

        image = Image.open(image_path).convert("RGB")
        depth = self.load_depth(depth_path)

        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.BICUBIC)
            depth_img = Image.fromarray(depth)
            depth = np.array(depth_img.resize((width, height), Image.NEAREST)).astype(np.float32)

        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        depth = torch.from_numpy(depth).float()
        valid_mask = torch.isfinite(depth)
        valid_mask &= depth > self.min_depth
        valid_mask &= depth < self.max_depth

        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "pixel_values": pixel_values,
            "depth": depth,
            "valid_mask": valid_mask.float(),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
    depths = torch.stack([x["depth"] for x in batch], dim=0)
    masks = torch.stack([x["valid_mask"] for x in batch], dim=0)

    return {
        "pixel_values": pixel_values,
        "depth": depths,
        "valid_mask": masks,
    }
