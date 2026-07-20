import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.ati_dataset_refactored import assign_motion_label
from dataset.dataset_utils import motion_measurements


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
DEPTH_EXTENSIONS = (".npy", ".png", ".tif", ".tiff", ".exr")
DEFAULT_AE_SCENE_PREFIX = "comlab_scene_ae"


@dataclass(frozen=True)
class SceneKey:
    light: str
    speed: str
    topology: str


@dataclass(frozen=True)
class AEFrameItem:
    rgb_path: Path
    depth_path: Path
    metadata_path: Path
    scene_name: str
    scene_prefix: str
    light: str
    collection_speed: str
    motion_speed: str
    topology: str
    lap_id: str
    frame_id: str
    linear_speed: float
    angular_speed: float
    acceleration: float


@dataclass
class MotionSequence:
    scene_name: str
    scene_prefix: str
    light: str
    speed: str
    topology: str
    lap_id: str
    features: np.ndarray
    frame_ids: List[str]
    exposure: Optional[float] = None
    gain: Optional[float] = None
    dataset_indices: Optional[List[int]] = None

    @property
    def scene_key(self) -> SceneKey:
        return SceneKey(self.light, self.speed, self.topology)

    @property
    def condition_key(self) -> Optional[Tuple[float, float]]:
        if self.exposure is None or self.gain is None:
            return None
        return float(self.exposure), float(self.gain)

    def __len__(self) -> int:
        return len(self.frame_ids)


def natural_key(value):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(value))
    ]


def topology_id(topology: str) -> str:
    topology = str(topology).strip()
    return topology if topology.startswith("topology") else f"topology{topology}"


def topology_number(topology: str) -> int:
    topology = topology_id(topology)
    suffix = topology[len("topology") :]
    if not suffix.isdigit():
        raise ValueError(f"Expected numeric topology name, got {topology}")
    return int(suffix)


def finite_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def log_normalize(value: float, min_value: float, max_value: float) -> float:
    if min_value <= 0.0 or max_value <= 0.0 or math.isclose(min_value, max_value):
        return 0.0

    log_value = math.log(max(value, min_value))
    log_min = math.log(min_value)
    log_max = math.log(max_value)
    return 2.0 * (log_value - log_min) / (log_max - log_min) - 1.0


def _yaw_from_vector(value) -> Optional[float]:
    if not isinstance(value, Mapping):
        return None
    yaw = finite_float(value.get("yaw_z", value.get("z")))
    if yaw is None:
        return None
    return abs(yaw)


def motion_features_from_metadata(metadata_path: Path, fallback) -> np.ndarray:
    linear_speed, angular_speed, acceleration = fallback
    imu_gyro = angular_speed
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return np.asarray(
            [linear_speed, angular_speed, imu_gyro, acceleration],
            dtype=np.float32,
        )

    imu = metadata.get("imu")
    if isinstance(imu, Mapping):
        parsed_gyro = _yaw_from_vector(imu.get("angular_velocity"))
        if parsed_gyro is not None:
            imu_gyro = parsed_gyro

    return np.asarray(
        [linear_speed, angular_speed, imu_gyro, acceleration],
        dtype=np.float32,
    )


def _parse_ae_scene_dir_name(
    name: str,
    scene_prefix: str,
    light_levels: Sequence[str],
) -> Optional[Tuple[str, str, str, str]]:
    prefix = f"{scene_prefix}_"
    if not name.startswith(prefix):
        return None

    parts = name[len(prefix) :].split("_")
    if len(parts) < 3:
        return None

    light = parts[0]
    topology = parts[-1]
    collection_speed = "_".join(parts[1:-1])
    if (
        light not in set(light_levels)
        or not collection_speed
        or not topology.startswith("topology")
    ):
        return None
    return scene_prefix, light, collection_speed, topology_id(topology)


def _index_files_by_stem(directory: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir(), key=lambda path: natural_key(path.name))
        if path.is_file() and path.suffix.lower() in extensions
    }


def _iter_lap_dirs(scene_dir: Path) -> List[Tuple[str, Path]]:
    if (scene_dir / "metadata").is_dir():
        return [(scene_dir.name, scene_dir)]

    lap_dirs = []
    for child in sorted(scene_dir.iterdir(), key=lambda path: natural_key(path.name)):
        if not child.is_dir():
            continue
        if child.name.startswith("lap_") and (child / "metadata").is_dir():
            lap_dirs.append((child.name, child))
            continue
        for lap_dir in sorted(child.iterdir(), key=lambda path: natural_key(path.name)):
            if (
                lap_dir.is_dir()
                and lap_dir.name.startswith("lap_")
                and (lap_dir / "metadata").is_dir()
            ):
                lap_dirs.append((f"{child.name}/{lap_dir.name}", lap_dir))
    return lap_dirs


class AutoExposureMotionDataset(Dataset):
    """
    Auto-exposure RGB/depth dataset for real-world ATI scenes.

    Expected layout:
        root/
          comlab_scene_ae_{light}_{collection_speed}_{topology}/
            pair_000_auto_exposure/
              lap_*/
                rgb/*.png
                depth/*.npy
                metadata/*.json
    """

    def __init__(
        self,
        root_dir: str,
        image_processor=None,
        image_size: Optional[Tuple[int, int]] = (518, 518),
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        min_valid_depth_ratio: float = 0.0,
        light_levels: Sequence[str] = ("dark", "dim", "normal"),
        speed_levels: Sequence[str] = ("stop", "slow", "fast", "rotate", "spin"),
        min_length: int = 1,
        scene_prefix: str = DEFAULT_AE_SCENE_PREFIX,
        ae_exposure: Optional[float] = None,
        ae_gain: Optional[float] = None,
    ):
        self.root_dir = Path(root_dir)
        self.image_processor = image_processor
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_valid_depth_ratio = min_valid_depth_ratio
        self.light_levels = tuple(light_levels)
        self.speed_levels = tuple(speed_levels)
        self.min_length = int(min_length)
        self.scene_prefix = scene_prefix
        self.ae_exposure = None if ae_exposure is None else float(ae_exposure)
        self.ae_gain = None if ae_gain is None else float(ae_gain)

        self.light_to_idx = {name: idx for idx, name in enumerate(self.light_levels)}
        self.speed_to_idx = {name: idx for idx, name in enumerate(self.speed_levels)}
        self.condition_names = (
            *[f"light_{name}" for name in self.light_levels],
            *[f"speed_{name}" for name in self.speed_levels],
            "exposure_log_norm",
            "gain_log_norm",
        )

        self.exposure_min = 1.0
        self.exposure_max = 1.0
        self.gain_min = 1.0
        self.gain_max = 1.0

        self.items, self.sequences = self._scan_items_and_sequences()

    @property
    def condition_dim(self) -> int:
        return len(self.condition_names)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        image = Image.open(item.rgb_path).convert("RGB")
        depth = self._load_depth(item.depth_path) / 1000.0

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
                    float(self.speed_to_idx[item.motion_speed]),
                    float(self.ae_exposure) if self.ae_exposure is not None else float("nan"),
                    float(self.ae_gain) if self.ae_gain is not None else float("nan"),
                    float(topology_number(item.topology)),
                ],
                dtype=torch.float32,
            ),
        )

    def sequences_by_scene(self) -> Dict[SceneKey, List[MotionSequence]]:
        grouped = defaultdict(list)
        for sequence in self.sequences:
            grouped[sequence.scene_key].append(sequence)
        return dict(grouped)

    def _scan_items_and_sequences(self):
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"AE root does not exist: {self.root_dir}")

        items = []
        sequence_rows = defaultdict(list)
        scan_stats = {
            "candidate_scenes": 0,
            "candidate_laps": 0,
            "included_laps": 0,
            "included_frames": 0,
            "missing_paired_files": 0,
            "missing_sensor_frames": 0,
            "invalid_metadata_frames": 0,
            "unknown_motion_frames": 0,
        }

        scene_dirs = sorted(self.root_dir.iterdir(), key=lambda path: natural_key(path.name))
        for scene_dir in tqdm(scene_dirs, desc="Scanning AE scenes"):
            if not scene_dir.is_dir():
                continue

            parsed_scene = _parse_ae_scene_dir_name(
                scene_dir.name,
                scene_prefix=self.scene_prefix,
                light_levels=self.light_levels,
            )
            if parsed_scene is None:
                continue

            scan_stats["candidate_scenes"] += 1
            scene_prefix, light, collection_speed, topology = parsed_scene

            for lap_id, lap_dir in _iter_lap_dirs(scene_dir):
                scan_stats["candidate_laps"] += 1
                rgb_files = _index_files_by_stem(lap_dir / "rgb", IMAGE_EXTENSIONS)
                depth_files = _index_files_by_stem(lap_dir / "depth", DEPTH_EXTENSIONS)
                metadata_files = _index_files_by_stem(lap_dir / "metadata", (".json",))

                frame_ids = set(rgb_files) & set(depth_files) & set(metadata_files)
                scan_stats["missing_paired_files"] += (
                    len(set(rgb_files) | set(depth_files) | set(metadata_files))
                    - len(frame_ids)
                )

                lap_indices = []
                for frame_id in sorted(frame_ids, key=natural_key):
                    metadata_path = metadata_files[frame_id]
                    try:
                        with metadata_path.open("r", encoding="utf-8") as handle:
                            metadata = json.load(handle)
                    except (OSError, json.JSONDecodeError):
                        scan_stats["invalid_metadata_frames"] += 1
                        continue

                    measurements = motion_measurements(metadata)
                    if measurements is None:
                        scan_stats["missing_sensor_frames"] += 1
                        continue

                    linear_speed, angular_speed, acceleration = measurements
                    motion_speed = assign_motion_label(
                        linear_speed,
                        acceleration,
                        angular_speed,
                    )
                    if motion_speed not in self.speed_to_idx:
                        scan_stats["unknown_motion_frames"] += 1
                        continue

                    item = AEFrameItem(
                        rgb_path=rgb_files[frame_id],
                        depth_path=depth_files[frame_id],
                        metadata_path=metadata_path,
                        scene_name=scene_dir.name,
                        scene_prefix=scene_prefix,
                        light=light,
                        collection_speed=collection_speed,
                        motion_speed=motion_speed,
                        topology=topology,
                        lap_id=lap_id,
                        frame_id=frame_id,
                        linear_speed=linear_speed,
                        angular_speed=angular_speed,
                        acceleration=acceleration,
                    )
                    item_idx = len(items)
                    items.append(item)
                    lap_indices.append(item_idx)

                if len(lap_indices) < self.min_length:
                    continue

                sequence_key = (
                    scene_dir.name,
                    scene_prefix,
                    light,
                    collection_speed,
                    topology,
                    lap_id,
                )
                sequence_rows[sequence_key].extend(lap_indices)
                scan_stats["included_laps"] += 1
                scan_stats["included_frames"] += len(lap_indices)

        sequences = []
        for (
            scene_name,
            scene_prefix,
            light,
            collection_speed,
            topology,
            lap_id,
        ), indices in sequence_rows.items():
            indices = sorted(indices, key=lambda index: natural_key(items[index].frame_id))
            features = np.stack(
                [
                    motion_features_from_metadata(
                        items[index].metadata_path,
                        fallback=(
                            items[index].linear_speed,
                            items[index].angular_speed,
                            items[index].acceleration,
                        ),
                    )
                    for index in indices
                ],
                axis=0,
            )
            sequences.append(
                MotionSequence(
                    scene_name=scene_name,
                    scene_prefix=scene_prefix,
                    light=light,
                    speed=collection_speed,
                    topology=topology,
                    lap_id=lap_id,
                    features=features,
                    frame_ids=[items[index].frame_id for index in indices],
                    dataset_indices=indices,
                )
            )

        self.scan_stats = scan_stats
        return items, sequences

    def _make_condition(self, item: AEFrameItem) -> torch.Tensor:
        values = [0.0] * self.condition_dim
        values[self.light_to_idx[item.light]] = 1.0

        speed_offset = len(self.light_levels)
        values[speed_offset + self.speed_to_idx[item.motion_speed]] = 1.0

        values[-2] = (
            log_normalize(self.ae_exposure, self.exposure_min, self.exposure_max)
            if self.ae_exposure is not None
            else 0.0
        )
        values[-1] = (
            log_normalize(self.ae_gain, self.gain_min, self.gain_max)
            if self.ae_gain is not None
            else 0.0
        )

        return torch.tensor(values, dtype=torch.float32)

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
