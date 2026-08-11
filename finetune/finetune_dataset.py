from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MDEFineTuningDataset(Dataset):
    """CSV에 매칭된 optimal(canonical) RGB/depth pair만 반환한다."""

    REQUIRED_COLUMNS = {
        "scene",
        "matched_rgb_path",
        "matched_depth_path",
        "match_status",
        "registration_status",
    }

    def __init__(
        self,
        csv_paths: str | Path | Sequence[str | Path],
        image_processor,
        camera_model_name: str,
        min_depth: float,
        max_depth: float,
        topologies: Sequence[str] | None = None,
        path_replacements: Mapping[str, str] | None = None,
        valid_match_status: str = "matched",
        valid_registration_status: str = "registered",
    ) -> None:
        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]

        frames = [pd.read_csv(path) for path in map(Path, csv_paths)]
        if not frames:
            raise ValueError("No CSV files were provided.")

        table = pd.concat(frames, ignore_index=True)
        missing = self.REQUIRED_COLUMNS - set(table.columns)
        if missing:
            raise ValueError("CSV is missing columns: " + ", ".join(sorted(missing)))

        table = table.loc[
            (table["match_status"] == valid_match_status)
            & (table["registration_status"] == valid_registration_status)
        ]
        if topologies:
            names = {
                name if str(name).startswith("topology") else f"topology{name}"
                for name in map(str, topologies)
            }
            table = table.loc[
                table["scene"].map(
                    lambda scene: any(part in names for part in str(scene).split("_"))
                )
            ]

        self.data = table.dropna(
            subset=["matched_rgb_path", "matched_depth_path"]
        ).drop_duplicates(
            subset=["matched_rgb_path", "matched_depth_path"]
        ).reset_index(drop=True)
        if self.data.empty:
            raise ValueError("No optimal RGB/depth pairs remain after filtering.")

        self.image_processor = image_processor
        self.depth_scale = 1000.0 if camera_model_name.startswith("Orbbec") else 1.0
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.path_replacements = dict(path_replacements or {})

    def __len__(self) -> int:
        return len(self.data)

    def _path(self, value: str) -> Path:
        value = str(value)
        for old, new in sorted(
            self.path_replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if value.startswith(old):
                value = new + value[len(old):]
                break
        return Path(value)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[index]
        with Image.open(self._path(row["matched_rgb_path"])) as image:
            rgb = image.convert("RGB")
            pixel_values = self.image_processor(
                images=rgb, return_tensors="pt"
            )["pixel_values"][0]

        depth = np.load(self._path(row["matched_depth_path"])).astype(np.float32)
        depth = np.squeeze(depth) / self.depth_scale
        depth = torch.from_numpy(depth)
        valid_mask = (
            torch.isfinite(depth)
            & (depth > self.min_depth)
            & (depth < self.max_depth)
        )
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))

        return {
            "pixel_values": pixel_values,
            "depth": depth,
            "valid_mask": valid_mask,
        }
