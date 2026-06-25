import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.dataset_utils import *

LIGHT_LEVELS = ("dark", "dim", "normal")
MOTION_LEVELS = ("stop", "slow", "fast", "rotate", "spin")
TRAIN_SCENE_PREFIX = "comlab_scene"
VALIDATION_SCENE_PREFIX = "val_comlab_scene"

V_STOP = 0.03
W_STOP = 0.05
V_SLOW = 0.25
V_FAST = 0.75
W_ROTATE = 0.5
W_SPIN = 1.0
A_SLOW = 0.5
A_FAST = 1.0

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


def assign_motion_label(
    v,
    a,
    w,
    v_stop=V_STOP,
    w_stop=W_STOP,
    w_rotate=W_ROTATE,
    w_spin=W_SPIN,
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

def _topology_number(topology: str) -> int:
    topology_suffix = topology[len("topology"):]
    if not topology_suffix.isdigit():
        raise ValueError(f"Expected numeric topology name, got {topology}")
    return int(topology_suffix)

class ATIRealWorldUncertaintyBaseDataset(Dataset):
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
        speed_levels: Sequence[str] = MOTION_LEVELS,
        spin_threshold: float = W_SPIN,
        max_samples: Optional[int] = None,
        topologies: Optional[Sequence[str]] = None,
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
        self.topologies = (
            tuple(dict.fromkeys(str(name).strip() for name in topologies))
            if topologies is not None
            else None
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
            topology_msg = (
                f" for topologies {list(self.topologies)}"
                if self.topologies is not None
                else ""
            )
            raise FileNotFoundError(
                f"No ATI RGB/depth pairs found in {self.split_name}{topology_msg} "
                f"under {self.root_dir}"
            )

        exposures = [item.exposure for item in all_items]
        gains = [item.gain for item in all_items]
        self.exposure_min = min(exposures)
        self.exposure_max = max(exposures)
        self.gain_min = min(gains)
        self.gain_max = max(gains)
        if max_samples is not None and max_samples > 0:
            all_items = all_items[:max_samples]

        self.items = all_items
    
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
        for scene_dir in tqdm(sorted(self.root_dir.iterdir()), desc="Scanning scenes"):
            if not scene_dir.is_dir():
                continue

            parsed_scene = parse_scene_dir_name(
                scene_dir.name,
                self.scene_prefix,
                self.light_levels,
            )
            if parsed_scene is None:
                continue
            scene_prefix, light, collection_speed, topology = parsed_scene
            if self.topologies is not None and topology not in self.topologies:
                continue
            
            for exposure_dir in sorted(scene_dir.iterdir()):
                if not exposure_dir.is_dir():
                    print(exposure_dir.name, "is not a directory. Skipping....")
                    continue
                
                parsed_exposure = parse_exposure_dir_name(exposure_dir.name)
                if parsed_exposure is None:
                    print(exposure_dir.name, "is not a valid exposure directory name. Skipping....")
                    continue

                exposure, gain = parsed_exposure
                for lap_dir in sorted(exposure_dir.iterdir()):
                    if (
                        not lap_dir.is_dir()
                        or re.fullmatch(r"lap_\d+", lap_dir.name) is None
                    ):
                        print(lap_dir.name, "is not a valid lap directory name. Skipping....")
                        continue

                    rgb_files = index_files_by_stem(lap_dir / "rgb", [".png"])
                    depth_files = index_files_by_stem(lap_dir / "depth", [".npy"])
                    metadata_files = index_files_by_stem(lap_dir / "metadata",[".json"])
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

                        measurements = motion_measurements(metadata)
                        if measurements is None:
                            scan_stats["missing_sensor_frames"] += 1
                            continue

                        linear_speed, angular_speed, acceleration = measurements
                        speed = assign_motion_label(
                            linear_speed,
                            acceleration,
                            angular_speed
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

    @property
    def condition_dim(self) -> int:
        return len(self.condition_names)

    def __len__(self):
        return len(self.items)

    def _load_depth(self, path: Path) -> np.ndarray:
        depth = np.load(path).astype(np.float32)
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

        values[-2] = log_normalize(item.exposure, self.exposure_min, self.exposure_max)
        values[-1] = log_normalize(item.gain, self.gain_min, self.gain_max)

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
                    float(_topology_number(item.topology)),
                ],
                dtype=torch.float32,
            ),
        )

class ATIRealWorldUncertaintyDataset(ATIRealWorldUncertaintyBaseDataset):
    """Training-only dataset backed by comlab_scene_* directories."""

    scene_prefix = TRAIN_SCENE_PREFIX
    split_name = "train"


class ATIRealWorldUncertaintyValidationDataset(ATIRealWorldUncertaintyBaseDataset):
    """Validation-only dataset backed by val_comlab_scene_* directories."""

    scene_prefix = VALIDATION_SCENE_PREFIX
    split_name = "validation"


def ati_collate_fn(batch: List[ATISample]) -> Optional[ATIBatch]:
    batch = [
        sample
        for sample in batch
        if sample[4][0].item() >= sample[4][1].item()
    ]

    if not batch:
        return None

    return (
        torch.stack([sample[0] for sample in batch], dim=0),
        torch.stack([sample[1] for sample in batch], dim=0),
        torch.stack([sample[2] for sample in batch], dim=0),
        torch.stack([sample[3] for sample in batch], dim=0),
        torch.stack([sample[4] for sample in batch], dim=0),
    )