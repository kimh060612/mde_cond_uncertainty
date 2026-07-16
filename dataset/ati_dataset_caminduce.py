from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


ImagePairTransform = Callable[
    [Image.Image, Image.Image, Mapping[str, Any]],
    tuple[torch.Tensor, torch.Tensor],
]

LIGHT_LEVELS = ("normal", "dim", "dark")
MOTION_LEVELS = ("fast", "slow", "stop", "rotate", "spin")


def _normalize_topology_name(topology: str) -> str:
    topology = str(topology).strip()
    return topology if topology.startswith("topology") else f"topology{topology}"


def _topology_number(topology: str) -> int:
    topology = _normalize_topology_name(topology)
    topology_suffix = topology[len("topology"):]
    if not topology_suffix.isdigit():
        raise ValueError(f"Expected numeric topology name, got {topology}")
    return int(topology_suffix)


def _topology_from_scene(scene: str) -> str:
    matches = [part for part in str(scene).split("_") if part.startswith("topology")]
    if len(matches) != 1:
        raise ValueError(f"Cannot infer topology from scene: {scene}")
    return _normalize_topology_name(matches[0])


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


def _validate_camera_parameter_normalization(
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

    _validate_camera_parameter_normalization(parameter_range, scale, output_range)
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
    ) -> None:
        """
        Args:
            size:
                (height, width). None이면 원본 크기를 유지합니다.
        """
        self.size = size

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
        if self.size is not None:
            height, width = self.size
            resize_size = (width, height)

            canonical_image = canonical_image.resize(resize_size, resample=Image.Resampling.BILINEAR)
            candidate_image = candidate_image.resize(resize_size, resample=Image.Resampling.BILINEAR)

        return (
            self._to_tensor(canonical_image),
            self._to_tensor(candidate_image),
        )


class FoundationCameraGroupedDataset(Dataset[dict[str, Any]]):
    """
    하나의 (foundation model, physical camera model) 조합을 위한 Dataset.

    이 Dataset은 서로 다른 physical camera model을 한 학습 집합 안에서
    섞어 일반화하는 용도가 아닙니다. 예를 들어:

        DepthAnythingV2 + Orbbec Gemini 336L
        DepthAnythingV2 + Intel RealSense D435
        DepthPro        + Orbbec Gemini 336L

    각각에 대해 Dataset과 camera-induced-error head를 따로 생성합니다.

    Grouping의 목적:
        동일한 canonical observation에 matching된 서로 다른 camera
        *settings* (exposure, gain)를 한 group에 묶어 ranking loss를 계산.

    camera_setting_id:
        (candidate exposure, candidate gain)

    이것은 physical camera ID가 아닙니다.

    기본 group:
        scene
        + canonical_pair_index
        + matched_lap_dir
        + matched_frame_index
        + source_motion_label

    반환 shape:
        canonical_images:     [K, C, H, W]
        candidate_images:     [K, C, H, W]
        camera_context:       [K, 10]
        candidate_exposure:   [K]
        candidate_gain:       [K]
        abs_rel_degradation:  [K]

    camera_context:
        [light one-hot, motion one-hot, normalized exposure, normalized gain].
        Canonical parameter는 model input에 사용하지 않음.
    """

    REQUIRED_COLUMNS = {
        "scene",
        "source_exposure",
        "source_gain",
        "source_motion_label",
        "source_rgb_path",
        "source_depth_path",
        "canonical_exposure",
        "canonical_gain",
        "canonical_pair_index",
        "matched_lap_dir",
        "matched_frame_index",
        "matched_rgb_path",
        "matched_depth_path",
        "source_metric_abs_rel",
        "canonical_metric_abs_rel",
        "performance_degradation_abs_rel",
        "performance_degradation_rmse",
        "match_status",
        "registration_status",
    }

    DEFAULT_GROUP_COLUMNS = (
        "scene",
        "canonical_pair_index",
        "matched_lap_dir",
        "matched_frame_index",
        "source_motion_label",
    )

    def __init__(
        self,
        csv_paths: str | Path | Sequence[str | Path],
        *,
        foundation_model_name: str,
        camera_model_name: str,
        parameter_range: CameraParameterRange,
        candidates_per_group: int = 4,
        group_columns: Sequence[str] | None = None,
        candidate_sampling: Literal[
            "random",
            "parameter_diverse",
        ] = "parameter_diverse",
        parameter_normalization: Literal[
            "linear",
            "log",
        ] = "linear",
        context_output_range: Literal[
            "zero_one",
            "minus_one_one",
        ] = "zero_one",
        clip_camera_context: bool = True,
        pair_transform: ImagePairTransform | None = None,
        path_replacements: Mapping[str, str] | None = None,
        include_canonical_setting_as_candidate: bool = False,
        load_images: bool = True,
        load_depth: bool = False,
        valid_match_status: str = "matched",
        valid_registration_status: str = "registered",
        min_overlap_ratio: float | None = None,
        min_ecc_score: float | None = None,
        max_time_diff_sec: float | None = None,
        topologies: Sequence[str] | None = None,
        validate_optional_pair_columns: bool = True,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if not foundation_model_name.strip():
            raise ValueError("foundation_model_name must be a non-empty string.")
        if not camera_model_name.strip():
            raise ValueError("camera_model_name must be a non-empty string.")
        if candidates_per_group < 2:
            raise ValueError("candidates_per_group must be at least 2.")
        if candidate_sampling not in {"random", "parameter_diverse"}:
            raise ValueError("candidate_sampling must be 'random' or 'parameter_diverse'.")

        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]

        csv_paths = [Path(path) for path in csv_paths]

        if not csv_paths:
            raise ValueError("At least one CSV path is required.")

        frames: list[pd.DataFrame] = []

        for csv_path in csv_paths:
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)

            frame = pd.read_csv(csv_path)
            frame["_csv_path"] = str(csv_path)
            frames.append(frame)

        self.min_depth = min_depth
        self.max_depth = max_depth
        table = pd.concat(frames, ignore_index=True)
        missing_columns = self.REQUIRED_COLUMNS - set(table.columns)
        if missing_columns:
            raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing_columns)))

        self.depth_scale = 1000 if camera_model_name.startswith("Orbbec") else 1.0
        _validate_camera_parameter_normalization(
            parameter_range,
            parameter_normalization,
            context_output_range,
        )

        self.foundation_model_name = foundation_model_name
        self.camera_model_name = camera_model_name
        self.parameter_range = parameter_range
        self.parameter_normalization = parameter_normalization
        self.context_output_range = context_output_range
        self.clip_camera_context = clip_camera_context
        self.light_levels = LIGHT_LEVELS
        self.motion_levels = MOTION_LEVELS
        self.topologies = tuple(dict.fromkeys(_normalize_topology_name(name) for name in topologies)) if topologies is not None else None
        self.light_to_idx = {name: idx for idx, name in enumerate(self.light_levels)}
        self.motion_to_idx = {name: idx for idx, name in enumerate(self.motion_levels)}
        self.camera_context_names = (
            *[f"light_{name}" for name in self.light_levels],
            *[f"motion_{name}" for name in self.motion_levels],
            "exposure_norm",
            "gain_norm",
        )

        # 향후 CSV에 pair 식별 column을 추가한 경우 실수로 다른
        # physical camera/foundation model 데이터를 섞는 것을 방지.
        if validate_optional_pair_columns:
            self._validate_optional_identity_column(
                table=table,
                column_name="foundation_model_name",
                expected_value=foundation_model_name,
            )
            self._validate_optional_identity_column(
                table=table,
                column_name="camera_model_name",
                expected_value=camera_model_name,
            )

        table = table.loc[~table["registration_status"].isin({"registration_failed", "failed"})].copy()
        table = table.loc[(table["match_status"] == valid_match_status) & (table["registration_status"] == valid_registration_status)].copy()

        if min_overlap_ratio is not None:
            if "registration_overlap_ratio" not in table:
                raise ValueError("registration_overlap_ratio is absent from CSV.")
            table = table.loc[table["registration_overlap_ratio"] >= min_overlap_ratio]

        if min_ecc_score is not None:
            if "registration_ecc_score" not in table:
                raise ValueError("registration_ecc_score is absent from CSV.")
            table = table.loc[table["registration_ecc_score"] >= min_ecc_score]

        if max_time_diff_sec is not None:
            if "time_diff_sec" not in table:
                raise ValueError("time_diff_sec is absent from CSV.")
            table = table.loc[table["time_diff_sec"].abs() <= max_time_diff_sec]

        if self.topologies is not None:
            topology_set = set(self.topologies)
            table = table.loc[table["scene"].map(lambda scene: _topology_from_scene(scene) in topology_set)]

        table = table.dropna(subset=["source_rgb_path", "matched_rgb_path", "matched_lap_dir", "matched_frame_index"]).reset_index(drop=True)

        self.group_columns = tuple(group_columns if group_columns is not None else self.DEFAULT_GROUP_COLUMNS)

        missing_group_columns = set(self.group_columns) - set(table.columns)
        if missing_group_columns:
            raise ValueError("Missing group columns: " + ", ".join(sorted(missing_group_columns)))

        self.candidates_per_group = candidates_per_group
        self.candidate_sampling = candidate_sampling
        self.pair_transform = pair_transform if pair_transform is not None else PairedResizeToTensor()
        self.path_replacements = dict(path_replacements or {})
        self.include_canonical_setting_as_candidate = include_canonical_setting_as_candidate
        self.load_images = load_images
        self.load_depth = load_depth
        self.seed = int(seed)
        self.epoch = 0

        # 'camera_id'라는 이름을 사용하지 않음.
        # 이것은 physical camera가 아니라 exposure/gain 설정 ID임.
        table["_camera_setting_id"] = list(
            zip(
                table["source_exposure"].astype(int),
                table["source_gain"].astype(int),
            )
        )

        table["_canonical_setting_id"] = list(
            zip(
                table["canonical_exposure"].astype(int),
                table["canonical_gain"].astype(int),
            )
        )

        if not include_canonical_setting_as_candidate:
            table = table.loc[table["_camera_setting_id"] != table["_canonical_setting_id"]].reset_index(drop=True)

        self.table = table

        # group -> camera setting -> row indices
        group_to_setting_rows: dict[
            tuple[Any, ...],
            dict[tuple[int, int], list[int]],
        ] = defaultdict(lambda: defaultdict(list))

        grouped = table.groupby(
            list(self.group_columns),
            sort=False,
            dropna=False,
        )

        for raw_group_key, group_frame in grouped:
            group_key = raw_group_key if isinstance(raw_group_key, tuple) else (raw_group_key,)

            canonical_path_count = group_frame["matched_rgb_path"].nunique()
            if canonical_path_count != 1:
                raise RuntimeError(f"Group {group_key} points to {canonical_path_count} canonical RGB paths.")

            canonical_setting_count = group_frame["_canonical_setting_id"].nunique()
            if canonical_setting_count != 1:
                raise RuntimeError(f"Group {group_key} contains {canonical_setting_count} canonical settings.")

            for row_index, row in group_frame.iterrows():
                setting_id = (int(row["source_exposure"]), int(row["source_gain"]))
                group_to_setting_rows[group_key][setting_id].append(int(row_index))

        self.group_to_setting_rows = {
            group_key: dict(setting_map)
            for group_key, setting_map
            in group_to_setting_rows.items()
            if len(setting_map) >= candidates_per_group
        }

        self.group_keys = list(self.group_to_setting_rows.keys())

        if not self.group_keys:
            raise ValueError(f"No eligible groups remain. Each canonical group needs at least {candidates_per_group} distinct exposure/gain settings.")

    @property
    def condition_dim(self) -> int:
        return len(self.camera_context_names)

    @staticmethod
    def _validate_optional_identity_column(
        *,
        table: pd.DataFrame,
        column_name: str,
        expected_value: str,
    ) -> None:
        if column_name not in table.columns:
            return

        values = {str(value) for value in table[column_name].dropna().unique()}

        if values and values != {expected_value}:
            raise ValueError(f"Column '{column_name}' contains {values}, but this Dataset is configured for '{expected_value}'. Do not mix different foundation-model/camera pairs.")

    @property
    def pair_name(self) -> str:
        return f"{self.foundation_model_name}__{self.camera_model_name}"

    def set_epoch(self, epoch: int) -> None:
        """
        매 epoch candidate sampling을 바꾸되 재현성을 유지합니다.

        persistent_workers=False인 DataLoader에서 epoch 시작 전에
        호출하는 것을 권장합니다.
        """
        self.epoch = int(epoch)

    def normalize_camera_parameters(
        self,
        exposure: torch.Tensor,
        gain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference에서도 동일하게 호출할 수 있는 public helper.

        Args:
            exposure:
                arbitrary shape
            gain:
                exposure와 같은 shape

        Returns:
            [*shape, 2]
        """
        return normalize_camera_parameters(
            exposure,
            gain,
            self.parameter_range,
            scale=self.parameter_normalization,
            output_range=self.context_output_range,
            clip=self.clip_camera_context,
        )

    def _remap_path(self, raw_path: str) -> Path:
        path_string = str(raw_path)

        for old_prefix, new_prefix in sorted(
            self.path_replacements.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if path_string.startswith(old_prefix):
                path_string = new_prefix + path_string[len(old_prefix):]
                break

        return Path(path_string)

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        if not path.is_file():
            raise FileNotFoundError(f"RGB image not found: {path}")

        with Image.open(path) as image:
            return image.convert("RGB").copy()

    @staticmethod
    def _load_depth(path: Path, depth_scale: float=1.0) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"Depth array not found: {path}")

        depth = np.load(path) / depth_scale
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return torch.from_numpy(np.asarray(depth, dtype=np.float32))

    def _make_rng(self, group_index: int) -> random.Random:
        mixed_seed = (self.seed + self.epoch * 1_000_003 + group_index * 9_176)
        return random.Random(mixed_seed)

    def _setting_to_normalized_array(
        self,
        setting_ids: list[tuple[int, int]],
    ) -> np.ndarray:
        exposures = torch.tensor([setting[0] for setting in setting_ids], dtype=torch.float32)
        gains = torch.tensor([setting[1] for setting in setting_ids], dtype=torch.float32)

        normalized = self.normalize_camera_parameters(exposures, gains)
        return normalized.cpu().numpy().astype(np.float64, copy=False)

    def _sample_random_settings(
        self,
        setting_ids: list[tuple[int, int]],
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        return rng.sample(setting_ids, self.candidates_per_group)

    def _sample_diverse_settings(
        self,
        setting_ids: list[tuple[int, int]],
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        """
        실제 model input과 같은 normalized exposure/gain 공간에서
        greedy farthest-point sampling을 수행합니다.
        """
        if len(setting_ids) == self.candidates_per_group:
            selected = setting_ids.copy()
            rng.shuffle(selected)
            return selected

        values = self._setting_to_normalized_array(setting_ids)
        first_index = rng.randrange(len(setting_ids))
        selected_indices = [first_index]

        while (len(selected_indices) < self.candidates_per_group):
            selected_points = values[selected_indices]

            pairwise_distances = np.linalg.norm(values[:, None, :] - selected_points[None, :, :], axis=-1)
            min_distance = pairwise_distances.min(axis=1)
            min_distance[selected_indices] = -1.0

            max_distance = min_distance.max()
            tied_indices = np.flatnonzero(np.isclose(min_distance, max_distance))

            next_index = int(rng.choice(tied_indices.tolist()))
            selected_indices.append(next_index)

        selected = [setting_ids[index] for index in selected_indices]
        rng.shuffle(selected)
        return selected

    def _select_setting_ids(
        self,
        setting_ids: list[tuple[int, int]],
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        if self.candidate_sampling == "random":
            return self._sample_random_settings(setting_ids, rng)

        return self._sample_diverse_settings(setting_ids, rng)

    @staticmethod
    def _row_metadata(
        row: pd.Series,
    ) -> dict[str, Any]:
        numeric_columns = [
            "time_diff_sec",
            "registration_overlap_ratio",
            "registration_ecc_score",
            "registration_dx_px",
            "registration_dy_px",
            "registered_depth_mean_abs_diff",
            "registered_depth_rmse_diff",
            "rgb_patch_x0",
            "rgb_patch_y0",
            "rgb_patch_width",
            "rgb_patch_height",
        ]

        result: dict[str, Any] = {}

        for column in numeric_columns:
            if column not in row.index:
                continue

            value = row[column]
            result[column] = float(value) if pd.notna(value) else float("nan")

        integer_columns = [
            "source_pair_index",
            "source_frame_index",
        ]

        for column in integer_columns:
            if column in row.index:
                result[column] = int(row[column])

        result["source_lap_dir"] = str(row["source_lap_dir"])
        result["source_motion_label"] = str(row["source_motion_label"])

        return result

    @staticmethod
    def _one_hot(
        labels: Sequence[str],
        label_to_idx: Mapping[str, int],
        label_name: str,
    ) -> torch.Tensor:
        values = torch.zeros((len(labels), len(label_to_idx)), dtype=torch.float32)
        for row_index, raw_label in enumerate(labels):
            label = str(raw_label).strip().lower()
            if label not in label_to_idx:
                raise ValueError(f"Unknown {label_name} label: {raw_label}")
            values[row_index, label_to_idx[label]] = 1.0
        return values

    def _row_light_label(self, row: pd.Series) -> str:
        for column in ("source_light", "light"):
            if column in row.index and pd.notna(row[column]):
                return str(row[column]).strip().lower()

        scene_tokens = str(row["scene"]).split("_")
        matches = [label for label in self.light_levels if label in scene_tokens]
        if len(matches) != 1 and not (scene_tokens[3] == "normal"):
            print(f"Scene tokens: {scene_tokens}, Light levels: {self.light_levels}")
            raise ValueError(f"Cannot infer light label from scene: {row['scene']}")
        return matches[0]

    def _make_camera_context(
        self,
        selected_rows: Sequence[pd.Series],
        exposures: torch.Tensor,
        gains: torch.Tensor,
    ) -> torch.Tensor:
        light_context = self._one_hot(
            [self._row_light_label(row) for row in selected_rows],
            self.light_to_idx,
            "light",
        )
        motion_context = self._one_hot(
            [row["source_motion_label"] for row in selected_rows],
            self.motion_to_idx,
            "motion",
        )
        parameter_context = self.normalize_camera_parameters(exposures, gains)
        return torch.cat([light_context, motion_context, parameter_context], dim=-1)

    def _make_info(self, row: pd.Series) -> torch.Tensor:
        light_label = self._row_light_label(row)
        motion_label = str(row["source_motion_label"]).strip().lower()
        if light_label not in self.light_to_idx:
            raise ValueError(f"Unknown light label: {light_label}")
        if motion_label not in self.motion_to_idx:
            raise ValueError(f"Unknown motion label: {row['source_motion_label']}")

        return torch.tensor(
            [
                1.0,
                0.0,
                float(self.light_to_idx[light_label]),
                float(self.motion_to_idx[motion_label]),
                float(row["canonical_exposure"]),
                float(row["canonical_gain"]),
                float(_topology_number(_topology_from_scene(row["scene"]))),
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        # 길이 단위는 image row가 아니라 canonical group.
        return len(self.group_keys)

    def __getitem__(self, group_index: int) -> dict[str, Any]:
        group_key = self.group_keys[group_index]
        setting_map = self.group_to_setting_rows[group_key]

        rng = self._make_rng(group_index)
        setting_ids = list(setting_map.keys())
        selected_setting_ids = self._select_setting_ids(setting_ids, rng)

        selected_rows: list[pd.Series] = []

        for setting_id in selected_setting_ids:
            row_indices = setting_map[setting_id]
            selected_row_index = rng.choice(row_indices)
            selected_rows.append(self.table.iloc[selected_row_index])

        reference_row = selected_rows[0]
        canonical_rgb_path = self._remap_path(reference_row["matched_rgb_path"])
        canonical_depth_path = self._remap_path(reference_row["matched_depth_path"])

        candidate_rgb_paths = [self._remap_path(row["source_rgb_path"]) for row in selected_rows]
        candidate_depth_paths = [self._remap_path(row["source_depth_path"]) for row in selected_rows]

        exposures = torch.tensor(
            [float(row["source_exposure"]) for row in selected_rows],
            dtype=torch.float32,
        )
        gains = torch.tensor(
            [float(row["source_gain"]) for row in selected_rows],
            dtype=torch.float32,
        )

        camera_context = self._make_camera_context(selected_rows, exposures, gains)

        canonical_exposure = float(reference_row["canonical_exposure"])
        canonical_gain = float(reference_row["canonical_gain"])
        candidate_abs_rel = torch.tensor(
            [float(row["source_metric_abs_rel"]) for row in selected_rows],
            dtype=torch.float32,
        )
        canonical_abs_rel = torch.tensor(
            [float(row["canonical_metric_abs_rel"]) for row in selected_rows],
            dtype=torch.float32,
        )
        abs_rel_degradation = torch.tensor(
            [float(row["performance_degradation_abs_rel"]) for row in selected_rows],
            dtype=torch.float32,
        )
        rmse_degradation = torch.tensor(
            [float(row["performance_degradation_rmse"]) for row in selected_rows],
            dtype=torch.float32,
        )

        result: dict[str, Any] = {
            "group_index": torch.tensor(group_index, dtype=torch.long),
            "group_key": "|".join(map(str, group_key)),
            "pair_name": self.pair_name,
            "foundation_model_name": self.foundation_model_name,
            "camera_model_name": self.camera_model_name,
            "scene": str(reference_row["scene"]),
            "candidate_exposure": exposures,
            "candidate_gain": gains,
            "candidate_abs_rel": candidate_abs_rel,
            "canonical_abs_rel": canonical_abs_rel,
            "abs_rel_degradation": abs_rel_degradation,
            "rmse_degradation": rmse_degradation,
            "camera_context": camera_context,
            "canonical_exposure": torch.tensor(canonical_exposure, dtype=torch.float32),
            "canonical_gain": torch.tensor(canonical_gain, dtype=torch.float32),
            "canonical_rgb_path": str(canonical_rgb_path),
            "candidate_rgb_paths": [str(path) for path in candidate_rgb_paths],
            "pair_metadata": [self._row_metadata(row) for row in selected_rows],
            "info": self._make_info(reference_row),
        }

        if not self.load_images:
            return result

        canonical_pil = self._load_rgb(canonical_rgb_path)

        canonical_tensors: list[torch.Tensor] = []
        candidate_tensors: list[torch.Tensor] = []

        for row, candidate_path in zip(selected_rows, candidate_rgb_paths):
            candidate_pil = self._load_rgb(candidate_path)

            canonical_tensor, candidate_tensor = self.pair_transform(
                canonical_pil.copy(),
                candidate_pil,
                row.to_dict(),
            )
            if not (canonical_tensor.shape == candidate_tensor.shape):
                raise RuntimeError(f"pair_transform must return canonical and candidate tensors with identical shapes. Got {canonical_tensor.shape} and {candidate_tensor.shape}.")

            canonical_tensors.append(canonical_tensor)
            candidate_tensors.append(candidate_tensor)

        result["canonical_images"] = torch.stack(canonical_tensors, dim=0)
        result["candidate_images"] = torch.stack(candidate_tensors, dim=0)

        if self.load_depth:
            canonical_depth = self._load_depth(canonical_depth_path, depth_scale=self.depth_scale)
            result["canonical_depths"] = canonical_depth.unsqueeze(0).expand(
                self.candidates_per_group, *canonical_depth.shape
            ).clone()

            result["candidate_depths"] = torch.stack(
                [self._load_depth(path, depth_scale=self.depth_scale) for path in candidate_depth_paths],
                dim=0,
            )
            depth = result["candidate_depths"]
            valid_mask = torch.isfinite(depth)
            valid_mask &= depth > self.min_depth
            valid_mask &= depth < self.max_depth
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            result["candidate_depths"] = depth
            result["candidate_valid_mask"] = valid_mask

        return result

    def statistics(self) -> GroupStatistics:
        setting_counts = [len(setting_map) for setting_map in self.group_to_setting_rows.values()]

        distinct_setting_count = len(set(self.table["_camera_setting_id"].tolist()))
        return GroupStatistics(
            num_rows=len(self.table),
            num_groups=len(self.group_keys),
            num_distinct_camera_settings=distinct_setting_count,
            min_settings_per_group=min(setting_counts),
            median_settings_per_group=float(np.median(setting_counts)),
            max_settings_per_group=max(setting_counts),
            foundation_model_name=self.foundation_model_name,
            camera_model_name=self.camera_model_name,
        )


def flatten_group_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """
    DataLoader output [G, K, ...]를 model 입력용 [G*K, ...]로 펼칩니다.

    Ranking loss 계산 시 model output을 다시:
        output.reshape(G, K, ...)
    로 복원하십시오.
    """
    if "candidate_images" not in batch:
        raise KeyError("batch does not contain candidate_images.")

    candidates = batch["candidate_images"]
    canonicals = batch["canonical_images"]

    if candidates.ndim != 5:
        raise ValueError("candidate_images must have shape [G, K, C, H, W].")

    if canonicals.shape != candidates.shape:
        raise ValueError("canonical_images and candidate_images must have the same shape.")

    num_groups, num_candidates = candidates.shape[:2]

    flattened = dict(batch)

    flattened["candidate_images"] = candidates.reshape(num_groups * num_candidates, *candidates.shape[2:])
    flattened["canonical_images"] = canonicals.reshape(num_groups * num_candidates, *canonicals.shape[2:])
    flattened["camera_context"] = batch["camera_context"].reshape(num_groups * num_candidates, -1)
    flattened["candidate_exposure"] = batch["candidate_exposure"].reshape(num_groups * num_candidates)
    flattened["candidate_gain"] = batch["candidate_gain"].reshape(num_groups * num_candidates)

    for key in (
        "candidate_depths",
        "candidate_valid_mask",
        "canonical_depths",
        "candidate_abs_rel",
        "canonical_abs_rel",
        "abs_rel_degradation",
        "rmse_degradation",
    ):
        if key in batch:
            flattened[key] = batch[key].reshape(num_groups * num_candidates, *batch[key].shape[2:])

    flattened["num_groups"] = num_groups
    flattened["num_candidates"] = num_candidates

    return flattened
