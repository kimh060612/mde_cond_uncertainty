from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import re
import math

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
