from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any, Literal
from collections.abc import Callable, Mapping, Sequence

import re
import math
import numpy as np
import torch
from PIL import Image


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
    
def parse_scene_dir_name(
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


def parse_exposure_dir_name(name: str):
    match = re.fullmatch(
        r"pair_\d+_exposure_([0-9.]+)_gain_([0-9.]+)",
        name,
    )
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))

def index_files_by_stem(directory: Path, extensions: Sequence[str]) -> Dict[str, Path]:
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


def motion_measurements(metadata: Mapping) -> Optional[Tuple[float, float, float]]:
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

def log_normalize(value: float, min_value: float, max_value: float) -> float:
    if min_value <= 0.0 or max_value <= 0.0 or math.isclose(min_value, max_value):
        return 0.0

    log_value = math.log(max(value, min_value))
    log_min = math.log(min_value)
    log_max = math.log(max_value)
    return 2.0 * (log_value - log_min) / (log_max - log_min) - 1.0


ImagePairTransform = Callable[
    [Image.Image, Image.Image, Mapping[str, Any]],
    tuple[torch.Tensor, torch.Tensor],
]

LIGHT_LEVELS = ("normal", "dim", "dark")
MOTION_LEVELS = ("fast", "slow", "stop", "rotate", "spin")


def normalize_topology_name(topology: str) -> str:
    topology = str(topology).strip()
    return topology if topology.startswith("topology") else f"topology{topology}"


def _topology_number(topology: str) -> int:
    topology = normalize_topology_name(topology)
    topology_suffix = topology[len("topology"):]
    if not topology_suffix.isdigit():
        raise ValueError(f"Expected numeric topology name, got {topology}")
    return int(topology_suffix)


def _topology_from_scene(scene: str) -> str:
    matches = [part for part in str(scene).split("_") if part.startswith("topology")]
    if len(matches) != 1:
        raise ValueError(f"Cannot infer topology from scene: {scene}")
    return normalize_topology_name(matches[0])


@dataclass(frozen=True)
class CameraParameterRange:
    """
    한 physical camera model에 대해 train/inference에서 공통으로 사용하는
    exposure/gain 허용 범위입니다.

    이 값은 dataset에서 자동 추정하기보다 camera API/실험 설계에서 정한
    고정 범위를 명시하는 것을 권장합니다.
    """
    exposure_min: float
    exposure_max: float
    gain_min: float
    gain_max: float

    def __post_init__(self) -> None:
        if not self.exposure_max > self.exposure_min:
            raise ValueError("exposure_max must be greater than exposure_min.")
        if not self.gain_max > self.gain_min:
            raise ValueError("gain_max must be greater than gain_min.")


def validate_camera_parameter_normalization(
    parameter_range: CameraParameterRange,
    scale: Literal["linear", "log"],
    output_range: Literal["zero_one", "minus_one_one"],
) -> None:
    if scale not in {"linear", "log"}:
        raise ValueError("scale must be 'linear' or 'log'.")
    if output_range not in {"zero_one", "minus_one_one"}:
        raise ValueError("output_range must be 'zero_one' or 'minus_one_one'.")
    if scale == "log":
        if parameter_range.exposure_min <= 0:
            raise ValueError("Log normalization requires exposure_min > 0.")
        if parameter_range.gain_min <= 0:
            raise ValueError("Log normalization requires gain_min > 0.")


def _normalize_camera_value(
    value: torch.Tensor,
    minimum: float,
    maximum: float,
    *,
    scale: Literal["linear", "log"],
    output_range: Literal["zero_one", "minus_one_one"],
    clip: bool,
) -> torch.Tensor:
    value = value.to(dtype=torch.float32)
    if scale == "log":
        value = torch.log(value.clamp_min(torch.finfo(value.dtype).tiny))
        minimum = math.log(minimum)
        maximum = math.log(maximum)

    normalized = (value - minimum) / (maximum - minimum)
    if clip:
        normalized = normalized.clamp(0.0, 1.0)
    if output_range == "minus_one_one":
        normalized = normalized.mul(2.0).sub(1.0)
    return normalized


def normalize_camera_parameters(
    exposure: torch.Tensor,
    gain: torch.Tensor,
    parameter_range: CameraParameterRange,
    *,
    scale: Literal["linear", "log"] = "linear",
    output_range: Literal["zero_one", "minus_one_one"] = "zero_one",
    clip: bool = True,
) -> torch.Tensor:
    if exposure.shape != gain.shape:
        raise ValueError("exposure and gain must have the same shape.")

    validate_camera_parameter_normalization(parameter_range, scale, output_range)
    exposure_norm = _normalize_camera_value(
        exposure,
        parameter_range.exposure_min,
        parameter_range.exposure_max,
        scale=scale,
        output_range=output_range,
        clip=clip,
    )
    gain_norm = _normalize_camera_value(
        gain,
        parameter_range.gain_min,
        parameter_range.gain_max,
        scale=scale,
        output_range=output_range,
        clip=clip,
    )
    return torch.stack([exposure_norm, gain_norm], dim=-1)


@dataclass(frozen=True)
class GroupStatistics:
    num_rows: int
    num_groups: int
    num_distinct_camera_settings: int
    min_settings_per_group: int
    median_settings_per_group: float
    max_settings_per_group: int
    foundation_model_name: str
    camera_model_name: str


class PairedResizeToTensor:
    """
    Canonical/candidate image에 동일한 deterministic resize를 적용합니다.

    Camera-induced appearance difference를 supervision으로 사용하므로
    candidate에만 color jitter를 적용하면 안 됩니다. Random spatial
    augmentation을 사용할 경우 canonical/candidate에 같은 parameter를
    적용하는 pair transform을 작성하십시오.
    """

    def __init__(
        self,
        size: tuple[int, int] | None = None,
        image_processor: Any | None = None,
    ) -> None:
        """
        Args:
            size:
                (height, width). None이면 원본 크기를 유지합니다.
            image_processor:
                CSV metric 생성에 사용한 Hugging Face image processor.
                지정하면 processor의 resize/rescale/normalize를 그대로 사용합니다.
        """
        self.size = size
        self.image_processor = image_processor

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)

        if array.ndim == 2:
            array = array[..., None]

        array = np.ascontiguousarray(array.transpose(2, 0, 1))
        return torch.from_numpy(array).div_(255.0)

    def __call__(
        self,
        canonical_image: Image.Image,
        candidate_image: Image.Image,
        _: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.image_processor is not None:
            pixel_values = self.image_processor(
                images=[canonical_image, candidate_image],
                return_tensors="pt",
            )["pixel_values"]
            return pixel_values[0], pixel_values[1]

        if self.size is not None:
            height, width = self.size
            resize_size = (width, height)

            canonical_image = canonical_image.resize(resize_size, resample=Image.Resampling.BILINEAR)
            candidate_image = candidate_image.resize(resize_size, resample=Image.Resampling.BILINEAR)

        return (
            self._to_tensor(canonical_image),
            self._to_tensor(candidate_image),
        )
