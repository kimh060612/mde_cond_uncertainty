import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SPEED_LEVELS = ("slow", "fast")
LIGHT_LEVELS = ("dark", "dim", "normal")
MOTION_LEVELS = ("stop", "slow", "fast", "rotate", "spin")
SCENE_PREFIXES = ("comlab_scene2", "realsense_scene")
TRAIN_SCENE_PREFIX = "comlab_scene"
VALIDATION_SCENE_PREFIX = "val_comlab_scene"
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
    scene_name: str
    scene_prefix: str
    light: str
    speed: str
    exposure: float
    gain: float
    frame_id: str
    topology: Optional[str] = None

def _topology_number(topology: str) -> int:
    topology_suffix = topology[len("topology"):]
    if not topology_suffix.isdigit():
        raise ValueError(f"Expected numeric topology name, got {topology}")
    return int(topology_suffix)

def _parse_scene_dir_name(name: str, scene_prefixes: Sequence[str]):
    for prefix in scene_prefixes:
        stem = f"{prefix}_"
        if not name.startswith(stem):
            continue

        parts = name[len(stem):].split("_", maxsplit=2)
        if len(parts) != 3:
            return None

        light, speed, topology = parts
        if not topology.startswith("topology"):
            return None
        return prefix, light, speed, topology
    return None


def parse_exposure_dir_name(name: str):
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


def _stable_split_items(
    items: List[ATIFrameItem],
    split: str,
    val_ratio: float,
    seed: int,
) -> List[ATIFrameItem]:
    if split in ("all", "full"):
        return items

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(items), generator=generator).tolist()

    val_count = max(1, int(round(len(items) * val_ratio)))
    val_indices = set(order[:val_count])

    if split in ("val", "validation"):
        return [item for idx, item in enumerate(items) if idx in val_indices]
    if split == "train":
        return [item for idx, item in enumerate(items) if idx not in val_indices]

    raise ValueError(f"Unsupported split: {split}")


def _log_normalize(value: float, min_value: float, max_value: float) -> float:
    if min_value <= 0.0 or max_value <= 0.0 or math.isclose(min_value, max_value):
        return 0.0

    log_value = math.log(max(value, min_value))
    log_min = math.log(min_value)
    log_max = math.log(max_value)
    return 2.0 * (log_value - log_min) / (log_max - log_min) - 1.0


class ATIRealWorldDepthDataset(Dataset):
    """
    Real-world ATI RGB/Realsense depth dataset.

    Expected layout:
        root/
          comlab_scene2_{light}_{speed}/
          realsense_scene_{light}_{speed}/
            exposure_{exposure}_gain_{gain}/
              rgb/{frame}.png
              depth/{frame}.npy

    Returned condition vector:
        [light one-hot, speed one-hot, normalized exposure, normalized gain]
    """
    scene_prefix = ""
    split_name = ""
    
    def __init__(
        self,
        root_dir: str = "/media/michael/ssd1/AIoT_ATI/realworld_dataset",
        image_processor=None,
        image_size: Optional[Tuple[int, int]] = (518, 518),
        split: str = "train",
        val_ratio: float = 0.2,
        split_seed: int = 42,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        min_valid_depth_ratio: float = 0.3,
        light_levels: Sequence[str] = LIGHT_LEVELS,
        speed_levels: Sequence[str] = MOTION_LEVELS,
        scene_prefixes: Sequence[str] = [],
        max_samples: Optional[int] = None,
        topologies: Optional[Sequence[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.image_processor = image_processor
        self.image_size = image_size
        self.split = self.split_name
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_valid_depth_ratio = min_valid_depth_ratio
        self.light_levels = tuple(light_levels)
        self.speed_levels = tuple(speed_levels)
        self.scene_prefixes = tuple([self.scene_prefix] + list(scene_prefixes))
        print(self.scene_prefix, self.scene_prefixes)
        self.topologies = tuple(topologies) if topologies is not None else None
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

        self.items = _stable_split_items(
            all_items,
            split=split,
            val_ratio=val_ratio,
            seed=split_seed,
        )

        if max_samples is not None and max_samples > 0:
            self.items = self.items[:max_samples]

        if not self.items:
            raise ValueError(f"Split '{split}' produced an empty ATI dataset")

    @property
    def condition_dim(self) -> int:
        return len(self.condition_names)

    def _scan_items(self) -> List[ATIFrameItem]:
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        items = []
        for scene_dir in sorted(self.root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            parsed_scene = _parse_scene_dir_name(scene_dir.name, self.scene_prefixes)
            if parsed_scene is None:
                continue

            scene_prefix, light, speed, topology = parsed_scene
            if self.topologies is not None and topology not in self.topologies:
                print(scene_dir.name, "does not match any known scene prefixes. Skipping....")
                continue
            
            if light not in self.light_to_idx or speed not in self.speed_to_idx:
                print(scene_dir.name, "does not match any known scene prefixes. Skipping....")
                continue

            for exposure_dir in sorted(scene_dir.iterdir()):
                if not exposure_dir.is_dir():
                    print(exposure_dir.name, "does not match any known scene prefixes. Skipping....")
                    continue

                parsed_exposure = parse_exposure_dir_name(exposure_dir.name)
                if parsed_exposure is None:
                    print(exposure_dir.name, "does not match any known scene prefixes. Skipping....")
                    continue

                exposure, gain = parsed_exposure
                for lap_dir in sorted(exposure_dir.iterdir()):
                    if (
                        not lap_dir.is_dir()
                        or re.fullmatch(r"lap_\d+", lap_dir.name) is None
                    ):
                        print(lap_dir.name, "does not match any known scene prefixes. Skipping....")
                        continue
                    
                    rgb_files = _index_files_by_stem(lap_dir / "rgb", [".png"])
                    depth_files = _index_files_by_stem(lap_dir / "depth", [".npy"])
                    paired_frame_ids = set(rgb_files) & set(depth_files)
                    
                    for frame_id in sorted(paired_frame_ids):
                        items.append(
                            ATIFrameItem(
                                rgb_path=rgb_files[frame_id],
                                depth_path=depth_files[frame_id],
                                scene_name=scene_dir.name,
                                scene_prefix=scene_prefix,
                                light=light,
                                speed=speed,
                                exposure=exposure,
                                gain=gain,
                                frame_id=frame_id,
                                topology=topology
                            )
                        )

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
                    float(_topology_number(item.topology)),
                ],
                dtype=torch.float32,
            ),
        )

class ATIRealWorldUncertaintyDataset(ATIRealWorldDepthDataset):
    """Training-only dataset backed by comlab_scene_* directories."""

    scene_prefix = TRAIN_SCENE_PREFIX
    split_name = "train"


class ATIRealWorldUncertaintyValidationDataset(ATIRealWorldDepthDataset):
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
