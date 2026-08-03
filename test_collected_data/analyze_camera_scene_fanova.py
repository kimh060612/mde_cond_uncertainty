#!/usr/bin/env python3
"""Camera–scene functional ANOVA on canonical-frame-match CSV files.

The script keeps the input CSV files read-only, aggregates temporally correlated
frames to laps, and writes statistical tables, diagnostics, plots, and reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationWarning,
)


REQUIRED_COLUMNS = {
    "scene",
    "match_status",
    "source_exposure",
    "source_gain",
    "source_lap_dir",
    "source_motion_label",
    "performance_degradation_abs_rel",
    "source_metric_abs_rel",
}
OPTIONAL_COLUMNS = {
    "source_pair_dir",
    "registration_overlap_ratio",
    "registration_ecc_score",
    "canonical_exposure",
    "canonical_gain",
}
READ_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
TARGETS = {
    "delta": "delta_abs_rel_mean",
    "raw": "raw_abs_rel_mean",
}
EPSILON = 1e-12


@dataclass
class AnalysisContext:
    """Shared output, argument, logger, and warning state."""

    args: argparse.Namespace
    output_dir: Path
    logger: logging.Logger
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        """Record a warning in both the logger and the final report."""
        self.warnings.append(message)
        self.logger.warning(message)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lap-level camera–scene functional ANOVA analysis."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--glob-pattern",
        default="**/*canonical_frame_matches*.csv",
    )
    parser.add_argument("--min-valid-frames", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--registration-overlap-threshold", type=float)
    parser.add_argument("--registration-ecc-threshold", type=float)
    args = parser.parse_args()
    if args.min_valid_frames < 1:
        parser.error("--min-valid-frames must be at least 1")
    if args.n_bootstrap < 1 and not args.skip_bootstrap:
        parser.error("--n-bootstrap must be at least 1")
    return args


def setup_context(args: argparse.Namespace) -> AnalysisContext:
    """Create output directories and configure file/console logging."""
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("camera_scene_fanova")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "analysis_warnings.log", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return AnalysisContext(args=args, output_dir=output_dir, logger=logger)


def finite_float(value: Any) -> float | None:
    """Return a JSON-safe finite float or None."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy, pandas, and Path values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document using safe scalar conversion."""
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """Compute a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_label(value: Any) -> str:
    """Format camera parameters without unnecessary decimal suffixes."""
    if pd.isna(value):
        return "unknown"
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"
    except (TypeError, ValueError):
        return str(value).strip()


def parse_scene_metadata(scene: Any, csv_path: Any) -> tuple[str, str, str, str]:
    """Parse light, speed, and topology from scene first, then filename."""
    original = "" if pd.isna(scene) else str(scene).strip()
    candidates = [original, Path(str(csv_path)).stem]
    pattern = re.compile(
        r"(?:^|_)(?P<light>dark|dim|normal|bright)"
        r"_(?P<speed>fast|normal|slow)"
        r"_(?P<topology>topology[A-Za-z0-9-]+)(?:_|$)",
        re.IGNORECASE,
    )
    for candidate in candidates:
        candidate = re.sub(r"\s*\(\d+\)(?=(?:_|$))", "", candidate)
        match = pattern.search(candidate)
        if match:
            light = match.group("light").lower()
            speed = match.group("speed").lower()
            topology = match.group("topology").lower()
            return f"{light}_{speed}_{topology}", topology, light, speed
    fallback = original or Path(str(csv_path)).stem
    return fallback, "unknown", "unknown", "unknown"


def discover_and_load(
    ctx: AnalysisContext,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Discover CSV files, exclude invalid schemas, and load selected columns."""
    input_root = ctx.args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root is not a directory: {input_root}")
    files = sorted(path.resolve() for path in input_root.glob(ctx.args.glob_pattern))
    ctx.logger.info("Discovered %d CSV files", len(files))
    if not files:
        raise FileNotFoundError(
            f"No CSV matched {ctx.args.glob_pattern!r} under {input_root}"
        )

    frames: list[pd.DataFrame] = []
    excluded: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path in files:
        file_meta = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        metadata.append(file_meta)
        try:
            header = pd.read_csv(path, nrows=0)
            missing = sorted(REQUIRED_COLUMNS - set(header.columns))
            if missing:
                excluded.append(
                    {
                        "source_csv_path": str(path),
                        "reason": "missing_required_columns",
                        "missing_columns": ";".join(missing),
                        "error": "",
                    }
                )
                inventory.append(
                    {
                        **file_meta,
                        "included": False,
                        "n_rows": np.nan,
                        "missing_columns": ";".join(missing),
                    }
                )
                ctx.warn(f"Excluded {path}: missing columns {missing}")
                continue
            selected = [col for col in header.columns if col in READ_COLUMNS]
            data = pd.read_csv(path, usecols=selected, low_memory=False)
            data["source_csv_path"] = str(path)
            frames.append(data)
            inventory.append(
                {
                    **file_meta,
                    "included": True,
                    "n_rows": len(data),
                    "missing_columns": "",
                }
            )
        except Exception as exc:  # keep partial analysis alive
            excluded.append(
                {
                    "source_csv_path": str(path),
                    "reason": "read_error",
                    "missing_columns": "",
                    "error": repr(exc),
                }
            )
            inventory.append(
                {
                    **file_meta,
                    "included": False,
                    "n_rows": np.nan,
                    "missing_columns": "",
                }
            )
            ctx.warn(f"Excluded unreadable file {path}: {exc}")

    excluded_df = pd.DataFrame(
        excluded,
        columns=["source_csv_path", "reason", "missing_columns", "error"],
    )
    inventory_df = pd.DataFrame(inventory)
    excluded_df.to_csv(ctx.output_dir / "excluded_files.csv", index=False)
    inventory_df.to_csv(ctx.output_dir / "dataset_inventory.csv", index=False)
    if not frames:
        raise RuntimeError("All discovered files were excluded; no data can be analyzed")

    frame_df = pd.concat(frames, ignore_index=True, sort=False)
    parsed = frame_df.apply(
        lambda row: parse_scene_metadata(row["scene"], row["source_csv_path"]),
        axis=1,
        result_type="expand",
    )
    parsed.columns = [
        "filename_scene_id",
        "topology",
        "light",
        "filename_speed",
    ]
    frame_df = pd.concat([frame_df, parsed], axis=1)
    frame_df["speed"] = (
        frame_df["source_motion_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "unknown", "nan": "unknown"})
    )
    frame_df["scene_id"] = (
        frame_df["light"].astype(str)
        + "_"
        + frame_df["speed"].astype(str)
        + "_"
        + frame_df["topology"].astype(str)
    )
    frame_df["camera_id"] = [
        f"exp_{value_label(exp)}_gain_{value_label(gain)}"
        for exp, gain in zip(
            frame_df["source_exposure"], frame_df["source_gain"], strict=False
        )
    ]
    for column in [
        "source_exposure",
        "source_gain",
        "performance_degradation_abs_rel",
        "source_metric_abs_rel",
        "registration_overlap_ratio",
        "registration_ecc_score",
    ]:
        if column in frame_df:
            frame_df[column] = pd.to_numeric(frame_df[column], errors="coerce")
    return frame_df, excluded_df, inventory_df, metadata


def add_validity_flags(frame_df: pd.DataFrame, ctx: AnalysisContext) -> pd.DataFrame:
    """Add matched, registration-quality, and target-validity flags."""
    data = frame_df.copy()
    data["matched_binary"] = (
        data["match_status"].astype(str).str.strip().str.lower() == "matched"
    ).astype(int)
    registration_ok = pd.Series(True, index=data.index)
    overlap_threshold = ctx.args.registration_overlap_threshold
    ecc_threshold = ctx.args.registration_ecc_threshold
    if overlap_threshold is not None:
        if "registration_overlap_ratio" not in data:
            ctx.warn(
                "Overlap threshold requested but registration_overlap_ratio is absent; "
                "all rows fail that optional quality condition."
            )
            registration_ok &= False
        else:
            registration_ok &= (
                data["registration_overlap_ratio"].notna()
                & (data["registration_overlap_ratio"] >= overlap_threshold)
            )
    if ecc_threshold is not None:
        if "registration_ecc_score" not in data:
            ctx.warn(
                "ECC threshold requested but registration_ecc_score is absent; "
                "all rows fail that optional quality condition."
            )
            registration_ok &= False
        else:
            registration_ok &= (
                data["registration_ecc_score"].notna()
                & (data["registration_ecc_score"] >= ecc_threshold)
            )
    delta = data["performance_degradation_abs_rel"]
    raw = data["source_metric_abs_rel"]
    data["valid_delta"] = (
        data["matched_binary"].eq(1)
        & registration_ok
        & delta.notna()
        & np.isfinite(delta)
        & delta.ge(0)
    )
    data["valid_raw"] = (
        data["matched_binary"].eq(1)
        & registration_ok
        & raw.notna()
        & np.isfinite(raw)
        & raw.ge(0)
    )
    return data


def _safe_std(series: pd.Series) -> float:
    """Return sample standard deviation, retaining NaN for singleton groups."""
    return float(series.std(ddof=1)) if len(series) > 1 else np.nan


def aggregate_laps(
    frame_df: pd.DataFrame, ctx: AnalysisContext
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate frame observations to independent lap-level analysis units."""
    group_columns = ["scene_id", "camera_id", "source_csv_path"]
    if "source_pair_dir" in frame_df.columns:
        group_columns.append("source_pair_dir")
    group_columns.append("source_lap_dir")
    metadata_columns = [
        "topology",
        "light",
        "speed",
        "source_exposure",
        "source_gain",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in frame_df.groupby(group_columns, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, key_values, strict=False))
        for column in metadata_columns:
            record[column] = group[column].iloc[0]
        delta = group.loc[
            group["valid_delta"], "performance_degradation_abs_rel"
        ].astype(float)
        raw = group.loc[group["valid_raw"], "source_metric_abs_rel"].astype(float)
        record.update(
            {
                "delta_abs_rel_mean": delta.mean(),
                "delta_abs_rel_median": delta.median(),
                "delta_abs_rel_std": _safe_std(delta),
                "raw_abs_rel_mean": raw.mean(),
                "raw_abs_rel_median": raw.median(),
                "raw_abs_rel_std": _safe_std(raw),
                "n_valid_delta_frames": int(len(delta)),
                "n_valid_raw_frames": int(len(raw)),
                "n_total_frames": int(len(group)),
                "match_rate": float(group["matched_binary"].mean()),
                "mean_registration_overlap": (
                    group["registration_overlap_ratio"].mean()
                    if "registration_overlap_ratio" in group
                    else np.nan
                ),
                "mean_registration_ecc": (
                    group["registration_ecc_score"].mean()
                    if "registration_ecc_score" in group
                    else np.nan
                ),
            }
        )
        records.append(record)
    laps = pd.DataFrame(records)
    laps["eligible_delta"] = (
        laps["n_valid_delta_frames"] >= ctx.args.min_valid_frames
    )
    laps["eligible_raw"] = laps["n_valid_raw_frames"] >= ctx.args.min_valid_frames
    minimum_motion_laps = max(2, 2 * laps["camera_id"].nunique())
    sparse_motion_levels: dict[str, set[str]] = {}
    for target, flag_column in [
        ("delta", "eligible_delta"),
        ("raw", "eligible_raw"),
    ]:
        counts = laps.loc[laps[flag_column]].groupby(
            "speed", observed=True
        ).size()
        sparse = set(
            counts.loc[counts < minimum_motion_laps].index.astype(str)
        )
        sparse_motion_levels[target] = sparse
        if sparse:
            laps.loc[laps["speed"].isin(sparse), flag_column] = False
            ctx.warn(
                f"{target} excluded sparse motion-label levels "
                f"{sorted(sparse)}: fewer than {minimum_motion_laps} eligible "
                "lap×motion units prevented estimable scene-camera inference."
            )
    exclusions: list[dict[str, Any]] = []
    id_columns = group_columns
    for target, count_column, flag_column in [
        ("delta", "n_valid_delta_frames", "eligible_delta"),
        ("raw", "n_valid_raw_frames", "eligible_raw"),
    ]:
        rejected = laps.loc[~laps[flag_column]]
        for _, row in rejected.iterrows():
            reason = (
                "insufficient_motion_level_coverage"
                if str(row["speed"]) in sparse_motion_levels[target]
                and int(row[count_column]) >= ctx.args.min_valid_frames
                else "insufficient_valid_frames"
            )
            exclusions.append(
                {
                    **{column: row[column] for column in id_columns},
                    "target": target,
                    "valid_frame_count": int(row[count_column]),
                    "min_valid_frames": ctx.args.min_valid_frames,
                    "reason": reason,
                }
            )
    excluded_laps = pd.DataFrame(
        exclusions,
        columns=[
            *id_columns,
            "target",
            "valid_frame_count",
            "min_valid_frames",
            "reason",
        ],
    )
    cleaned = laps.loc[laps["eligible_delta"] | laps["eligible_raw"]].copy()
    cleaned.to_csv(ctx.output_dir / "cleaned_lap_level.csv", index=False)
    excluded_laps.to_csv(ctx.output_dir / "excluded_laps.csv", index=False)
    ctx.logger.info(
        "Aggregated %d lap units; %d retained for at least one target",
        len(laps),
        len(cleaned),
    )
    return cleaned, excluded_laps


def summarize_series(series: pd.Series) -> dict[str, float | int | None]:
    """Compute robust descriptive statistics for a numeric series."""
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "median": float(values.median()),
        "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def registration_match_rates(
    frame_df: pd.DataFrame, ctx: AnalysisContext
) -> pd.DataFrame:
    """Calculate frame-level match rates for requested factors and fit a GLM."""
    tables: list[pd.DataFrame] = []
    group_specs = {
        "scene_camera": ["scene_id", "camera_id"],
        "scene": ["scene_id"],
        "camera": ["camera_id"],
        "topology": ["topology"],
        "light": ["light"],
        "speed": ["speed"],
    }
    for grouping, columns in group_specs.items():
        table = (
            frame_df.groupby(columns, dropna=False)["matched_binary"]
            .agg(n_frames="size", n_matched="sum", match_rate="mean")
            .reset_index()
        )
        table.insert(0, "grouping", grouping)
        for column in ["scene_id", "camera_id", "topology", "light", "speed"]:
            if column not in table:
                table[column] = ""
        tables.append(table)
    result = pd.concat(tables, ignore_index=True, sort=False)
    result.to_csv(
        ctx.output_dir / "registration_match_rate_analysis.csv", index=False
    )

    glm_path = ctx.output_dir / "registration_glm_summary.txt"
    try:
        cell = (
            frame_df.groupby(["scene_id", "camera_id"], observed=True)
            ["matched_binary"]
            .agg(["sum", "count"])
            .reset_index()
        )
        cell["proportion"] = cell["sum"] / cell["count"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            glm = smf.glm(
                "proportion ~ C(scene_id, Sum) * C(camera_id, Sum)",
                data=cell,
                family=sm.families.Binomial(),
                freq_weights=cell["count"],
            ).fit(maxiter=100)
        diagnostic_warnings = [
            str(item.message)
            for item in caught
            if issubclass(
                item.category,
                (PerfectSeparationWarning, ConvergenceWarning, RuntimeWarning),
            )
        ]
        if diagnostic_warnings:
            ctx.warn(
                "Registration GLM emitted convergence/separation diagnostics: "
                + " | ".join(sorted(set(diagnostic_warnings)))
            )
        glm_path.write_text(
            "Binomial GLM fit to scene-camera binomial counts; frequency weights "
            "are algebraically equivalent to frame-level Bernoulli replication.\n"
            + (
                "Diagnostics: " + " | ".join(sorted(set(diagnostic_warnings))) + "\n"
                if diagnostic_warnings
                else "Diagnostics: none recorded.\n"
            )
            + "\n"
            + glm.summary().as_text(),
            encoding="utf-8",
        )
    except Exception as exc:
        ctx.warn(f"Registration binomial GLM failed; rate tables retained: {exc}")
        glm_path.write_text(f"GLM fitting failed: {exc}\n", encoding="utf-8")
    return result


def dataset_quality(
    frame_df: pd.DataFrame,
    lap_df: pd.DataFrame,
    excluded_files: pd.DataFrame,
    inventory: pd.DataFrame,
    match_rates: pd.DataFrame,
    ctx: AnalysisContext,
) -> dict[str, Any]:
    """Write balance/missing-cell tables and a machine-readable quality summary."""
    cell = (
        lap_df.groupby(["scene_id", "camera_id"], observed=True)
        .agg(
            n_laps=("source_lap_dir", "size"),
            n_delta_laps=("eligible_delta", "sum"),
            n_raw_laps=("eligible_raw", "sum"),
            mean_match_rate=("match_rate", "mean"),
        )
        .reset_index()
    )
    all_scenes = sorted(frame_df["scene_id"].dropna().astype(str).unique())
    all_cameras = sorted(frame_df["camera_id"].dropna().astype(str).unique())
    full = pd.MultiIndex.from_product(
        [all_scenes, all_cameras], names=["scene_id", "camera_id"]
    ).to_frame(index=False)
    balance = full.merge(cell, how="left", on=["scene_id", "camera_id"])
    for column in ["n_laps", "n_delta_laps", "n_raw_laps"]:
        balance[column] = balance[column].fillna(0).astype(int)
    balance.to_csv(ctx.output_dir / "scene_camera_balance.csv", index=False)
    missing = balance.loc[balance["n_laps"].eq(0), ["scene_id", "camera_id"]]
    missing.to_csv(ctx.output_dir / "missing_scene_camera_cells.csv", index=False)
    match_cell = match_rates.loc[
        match_rates["grouping"].eq("scene_camera"),
        ["scene_id", "camera_id", "n_frames", "n_matched", "match_rate"],
    ]
    match_cell.to_csv(ctx.output_dir / "matching_rates.csv", index=False)

    included_hashes = inventory.loc[inventory["included"], "sha256"]
    duplicated_hashes = sorted(
        included_hashes[included_hashes.duplicated(keep=False)].unique().tolist()
    )
    duplicate_columns = [
        column
        for column in [
            "scene_id",
            "camera_id",
            "source_pair_dir",
            "source_lap_dir",
            "performance_degradation_abs_rel",
            "source_metric_abs_rel",
            "match_status",
            "source_csv_path",
        ]
        if column in frame_df
    ]
    duplicate_rows = int(frame_df.duplicated(subset=duplicate_columns).sum())
    counts = cell["n_laps"]
    balanced = bool(
        not cell.empty
        and len(missing) == 0
        and counts.nunique(dropna=False) == 1
    )
    if not balanced:
        ctx.warn(
            "The scene-camera design is unbalanced; Type III/Sum-contrast models "
            "and both cell-balanced and observed-weighted decompositions are reported."
        )
    if duplicated_hashes or duplicate_rows:
        ctx.warn(
            f"Possible duplicates detected: {len(duplicated_hashes)} repeated file "
            f"hashes and {duplicate_rows} repeated analysis-key rows."
        )
    summary = {
        "n_csv_discovered": int(len(inventory)),
        "n_csv_included": int(inventory["included"].sum()),
        "n_csv_excluded": int(len(excluded_files)),
        "n_frames": int(len(frame_df)),
        "n_matched_rows": int(frame_df["matched_binary"].sum()),
        "n_registration_failed_rows": int(
            frame_df["match_status"]
            .astype(str)
            .str.lower()
            .eq("registration_failed")
            .sum()
        ),
        "n_scenes": int(frame_df["scene_id"].nunique()),
        "n_topologies": int(frame_df["topology"].nunique()),
        "n_lights": int(frame_df["light"].nunique()),
        "n_speeds": int(frame_df["speed"].nunique()),
        "n_camera_settings": int(frame_df["camera_id"].nunique()),
        "n_laps_retained_any_target": int(len(lap_df)),
        "n_delta_scenes": int(
            lap_df.loc[lap_df["eligible_delta"], "scene_id"].nunique()
        ),
        "n_raw_scenes": int(
            lap_df.loc[lap_df["eligible_raw"], "scene_id"].nunique()
        ),
        "delta_motion_levels": sorted(
            lap_df.loc[lap_df["eligible_delta"], "speed"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "raw_motion_levels": sorted(
            lap_df.loc[lap_df["eligible_raw"], "speed"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "scene_camera_lap_count_min": int(counts.min()) if len(counts) else 0,
        "scene_camera_lap_count_max": int(counts.max()) if len(counts) else 0,
        "n_missing_scene_camera_cells": int(len(missing)),
        "balanced": balanced,
        "exposure_levels": sorted(
            frame_df["source_exposure"].dropna().unique().tolist()
        ),
        "gain_levels": sorted(frame_df["source_gain"].dropna().unique().tolist()),
        "target_statistics_lap_level": {
            "delta_abs_rel_mean": summarize_series(
                lap_df.loc[lap_df["eligible_delta"], "delta_abs_rel_mean"]
            ),
            "raw_abs_rel_mean": summarize_series(
                lap_df.loc[lap_df["eligible_raw"], "raw_abs_rel_mean"]
            ),
        },
        "duplicate_file_hashes": duplicated_hashes,
        "possible_duplicate_rows": duplicate_rows,
    }
    write_json(ctx.output_dir / "data_quality_summary.json", summary)
    return summary


def effect_name(term: str) -> str:
    """Map statsmodels/Patsy term text to report-friendly effect labels."""
    if term == "Intercept":
        return "intercept"
    if term == "Residual":
        return "residual"
    if "scene_id" in term and "camera_id" in term:
        return "scene_camera"
    if "scene_id" in term:
        return "scene"
    if "topology" in term and "camera_id" in term:
        return "topology_camera"
    if "light" in term and "camera_id" in term:
        return "light_camera"
    if "speed" in term and "camera_id" in term:
        return "speed_camera"
    if "camera_id" in term:
        return "camera"
    if "topology" in term:
        return "topology"
    if "light" in term:
        return "light"
    if "speed" in term:
        return "speed"
    return term


def enrich_anova(table: pd.DataFrame) -> pd.DataFrame:
    """Add mean squares, standardized effect sizes, and significance labels."""
    result = table.copy()
    result.index.name = "term"
    result = result.reset_index()
    result["effect"] = result["term"].map(effect_name)
    result = result.rename(columns={"PR(>F)": "p_value"})
    for column in ["sum_sq", "df", "F", "p_value"]:
        if column not in result:
            result[column] = np.nan
    residual_rows = result["effect"].eq("residual")
    residual_ss = (
        float(result.loc[residual_rows, "sum_sq"].iloc[0])
        if residual_rows.any()
        else np.nan
    )
    residual_df = (
        float(result.loc[residual_rows, "df"].iloc[0])
        if residual_rows.any()
        else np.nan
    )
    mse = residual_ss / residual_df if residual_df > 0 else np.nan
    non_intercept = ~result["effect"].isin(["intercept"])
    total_ss = float(result.loc[non_intercept, "sum_sq"].sum())
    result["mean_sq"] = result["sum_sq"] / result["df"]
    result["eta_squared"] = np.where(
        result["effect"].isin(["intercept", "residual"]) | (total_ss <= 0),
        np.nan,
        result["sum_sq"] / total_ss,
    )
    result["partial_eta_squared"] = np.where(
        result["effect"].isin(["intercept", "residual"])
        | ((result["sum_sq"] + residual_ss) <= 0),
        np.nan,
        result["sum_sq"] / (result["sum_sq"] + residual_ss),
    )
    result["omega_squared"] = np.where(
        result["effect"].isin(["intercept", "residual"]) | (total_ss + mse <= 0),
        np.nan,
        (result["sum_sq"] - result["df"] * mse) / (total_ss + mse),
    )
    result["significance"] = result["p_value"].map(significance_label)
    return result[
        [
            "effect",
            "term",
            "df",
            "sum_sq",
            "mean_sq",
            "F",
            "p_value",
            "eta_squared",
            "partial_eta_squared",
            "omega_squared",
            "significance",
        ]
    ]


def significance_label(p_value: Any) -> str:
    """Return a conventional significance label without overstating results."""
    p = finite_float(p_value)
    if p is None:
        return "not_tested"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return "p<0.01"
    if p < 0.05:
        return "p<0.05"
    return "not_significant"


def target_laps(lap_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return finite, sufficiently sampled lap rows for a target."""
    column = TARGETS[target]
    flag = f"eligible_{target}"
    data = lap_df.loc[lap_df[flag]].copy()
    data = data.loc[np.isfinite(pd.to_numeric(data[column], errors="coerce"))]
    return data


def fit_primary_anova(
    lap_df: pd.DataFrame, target: str, ctx: AnalysisContext
) -> tuple[pd.DataFrame, Any | None, dict[str, float]]:
    """Fit the primary Sum-contrast scene-by-camera Type III ANOVA."""
    data = target_laps(lap_df, target)
    formula = (
        f"{TARGETS[target]} ~ C(scene_id, Sum) * C(camera_id, Sum)"
    )
    output_csv = ctx.output_dir / f"anova_scene_camera_{target}.csv"
    summary_path = ctx.output_dir / f"model_fit_summary_{target}.txt"
    try:
        model = smf.ols(formula, data=data).fit()
        table = enrich_anova(anova_lm(model, typ=3))
        table.to_csv(output_csv, index=False)
        summary_path.write_text(
            f"Formula: {formula}\n"
            f"N laps: {int(model.nobs)}\n"
            f"R-squared: {model.rsquared:.12g}\n"
            f"Adjusted R-squared: {model.rsquared_adj:.12g}\n"
            f"Design rank: {np.linalg.matrix_rank(model.model.exog)} / "
            f"{model.model.exog.shape[1]}\n\n"
            + model.summary().as_text(),
            encoding="utf-8",
        )
        fit = {
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "nobs": float(model.nobs),
        }
        return table, model, fit
    except Exception as exc:
        ctx.warn(f"Primary {target} ANOVA failed: {exc}")
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False)
        summary_path.write_text(f"Model fitting failed: {exc}\n", encoding="utf-8")
        return empty, None, {
            "r_squared": np.nan,
            "adjusted_r_squared": np.nan,
            "nobs": 0,
        }


def _fit_formula_rank(formula: str, data: pd.DataFrame) -> tuple[Any, int, int]:
    """Fit OLS and return model, design rank, and column count."""
    model = smf.ols(formula, data=data).fit()
    exog = model.model.exog
    return model, int(np.linalg.matrix_rank(exog)), int(exog.shape[1])


def fit_decomposed_anova(
    lap_df: pd.DataFrame, target: str, ctx: AnalysisContext
) -> tuple[pd.DataFrame, str]:
    """Fit topology/light/speed camera diagnostics, reducing aliased terms."""
    data = target_laps(lap_df, target)
    active_factors = [
        factor
        for factor in ["topology", "light", "speed"]
        if data[factor].nunique(dropna=True) > 1
    ]
    excluded_singletons = [
        factor
        for factor in ["topology", "light", "speed"]
        if factor not in active_factors
    ]
    if excluded_singletons:
        ctx.warn(
            f"{target} decomposed ANOVA excluded one-level factors: "
            f"{', '.join(excluded_singletons)}"
        )
    main_terms = [f"C({factor}, Sum)" for factor in active_factors]
    if data["camera_id"].nunique() > 1:
        main_terms.append("C(camera_id, Sum)")
    interaction_terms = [
        f"C({factor}, Sum):C(camera_id, Sum)" for factor in active_factors
    ]
    terms = main_terms + interaction_terms
    output = ctx.output_dir / f"anova_decomposed_{target}.csv"
    if not terms:
        ctx.warn(f"No multi-level factor is available for {target} diagnostics")
        pd.DataFrame().to_csv(output, index=False)
        return pd.DataFrame(), ""

    removed: list[str] = []
    while terms:
        formula = f"{TARGETS[target]} ~ " + " + ".join(terms)
        try:
            model, rank, columns = _fit_formula_rank(formula, data)
        except Exception as exc:
            removed_term = terms.pop()
            removed.append(removed_term)
            ctx.warn(
                f"{target} diagnostic term {removed_term} removed after fit "
                f"failure: {exc}"
            )
            continue
        if rank == columns:
            try:
                table = enrich_anova(anova_lm(model, typ=3))
                table["model_formula"] = formula
                table["design_rank"] = rank
                table["design_columns"] = columns
                table["removed_terms"] = ";".join(removed)
                table.to_csv(output, index=False)
                return table, formula
            except Exception as exc:
                removable = next(
                    (
                        term
                        for term in reversed(terms)
                        if ":C(camera_id, Sum)" in term
                    ),
                    terms[-1],
                )
                terms.remove(removable)
                removed.append(removable)
                ctx.warn(
                    f"{target} diagnostic term {removable} was not estimable "
                    f"under Type III ANOVA and was removed: {exc}"
                )
                continue
        removable = next(
            (
                term
                for term in reversed(terms)
                if ":C(camera_id, Sum)" in term
            ),
            terms[-1],
        )
        terms.remove(removable)
        removed.append(removable)
        ctx.warn(
            f"{target} decomposed design was rank deficient ({rank}/{columns}); "
            f"removed term {removable}."
        )
    pd.DataFrame().to_csv(output, index=False)
    ctx.warn(f"All decomposed {target} terms were non-estimable")
    return pd.DataFrame(), ""


def functional_decomposition(
    lap_df: pd.DataFrame,
    target: str,
    weighting: str,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Empirically decompose cell means into scene, camera, and interaction."""
    data = target_laps(lap_df, target)
    value = TARGETS[target]
    cell = (
        data.groupby(["scene_id", "camera_id"], observed=True)[value]
        .agg(y="mean", n_laps="size")
        .reset_index()
    )
    if cell.empty:
        raise ValueError(f"No eligible {target} cells")
    if weighting not in {"cell_balanced", "observed_weighted"}:
        raise ValueError(f"Unknown weighting: {weighting}")
    cell["weight"] = 1.0 if weighting == "cell_balanced" else cell["n_laps"].astype(float)
    f0 = float(np.average(cell["y"], weights=cell["weight"]))

    scene_rows: list[dict[str, Any]] = []
    for scene, group in cell.groupby("scene_id", observed=True):
        weights = np.ones(len(group)) if weighting == "cell_balanced" else group["weight"]
        mean = float(np.average(group["y"], weights=weights))
        scene_rows.append(
            {
                "target": target,
                "weighting": weighting,
                "scene_id": scene,
                "cell_mean": mean,
                "f_scene": mean - f0,
                "weight": float(np.sum(weights)),
            }
        )
    camera_rows: list[dict[str, Any]] = []
    for camera, group in cell.groupby("camera_id", observed=True):
        weights = np.ones(len(group)) if weighting == "cell_balanced" else group["weight"]
        mean = float(np.average(group["y"], weights=weights))
        camera_rows.append(
            {
                "target": target,
                "weighting": weighting,
                "camera_id": camera,
                "cell_mean": mean,
                "f_camera": mean - f0,
                "weight": float(np.sum(weights)),
            }
        )
    scene_effects = pd.DataFrame(scene_rows)
    camera_effects = pd.DataFrame(camera_rows)
    interaction = cell.merge(
        scene_effects[["scene_id", "f_scene"]], on="scene_id"
    ).merge(camera_effects[["camera_id", "f_camera"]], on="camera_id")
    interaction["f0"] = f0
    interaction["f_scene_camera"] = (
        interaction["y"]
        - f0
        - interaction["f_scene"]
        - interaction["f_camera"]
    )
    interaction.insert(0, "weighting", weighting)
    interaction.insert(0, "target", target)

    v_scene = float(
        np.average(
            interaction["f_scene"] ** 2,
            weights=interaction["weight"],
        )
    )
    v_camera = float(
        np.average(
            interaction["f_camera"] ** 2,
            weights=interaction["weight"],
        )
    )
    v_interaction = float(
        np.average(
            interaction["f_scene_camera"] ** 2,
            weights=interaction["weight"],
        )
    )
    cell_keys = ["scene_id", "camera_id"]
    lap_residual = data.merge(
        cell[cell_keys + ["y", "n_laps"]],
        on=cell_keys,
        how="inner",
        suffixes=("", "_cell"),
    )
    lap_residual["squared_residual"] = (
        lap_residual[value] - lap_residual["y"]
    ) ** 2
    if weighting == "observed_weighted":
        v_residual = float(lap_residual["squared_residual"].mean())
    else:
        per_cell = lap_residual.groupby(cell_keys, observed=True)[
            "squared_residual"
        ].mean()
        v_residual = float(per_cell.mean())
    component_sum = v_scene + v_camera + v_interaction + v_residual
    fractions = (
        [v_scene / component_sum, v_camera / component_sum,
         v_interaction / component_sum, v_residual / component_sum]
        if component_sum > EPSILON
        else [np.nan] * 4
    )
    variance = {
        "target": target,
        "weighting": weighting,
        "f0": f0,
        "V_scene": v_scene,
        "V_camera": v_camera,
        "V_scene_camera": v_interaction,
        "V_residual": v_residual,
        "V_component_sum": component_sum,
        "fraction_scene": fractions[0],
        "fraction_camera": fractions[1],
        "fraction_scene_camera": fractions[2],
        "fraction_residual": fractions[3],
        "n_cells": int(len(cell)),
        "n_laps": int(len(data)),
    }
    return variance, scene_effects, camera_effects, interaction


def run_functional_anova(
    lap_df: pd.DataFrame, ctx: AnalysisContext
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both functional decompositions for both targets and save outputs."""
    variances: list[dict[str, Any]] = []
    scene_effects: list[pd.DataFrame] = []
    camera_effects: list[pd.DataFrame] = []
    interactions: list[pd.DataFrame] = []
    for target in TARGETS:
        for weighting in ["cell_balanced", "observed_weighted"]:
            try:
                variance, scene, camera, interaction = functional_decomposition(
                    lap_df, target, weighting
                )
                variances.append(variance)
                scene_effects.append(scene)
                camera_effects.append(camera)
                interactions.append(interaction)
            except Exception as exc:
                ctx.warn(
                    f"Functional ANOVA failed for {target}/{weighting}: {exc}"
                )
    variance_df = pd.DataFrame(variances)
    scene_df = pd.concat(scene_effects, ignore_index=True) if scene_effects else pd.DataFrame()
    camera_df = (
        pd.concat(camera_effects, ignore_index=True) if camera_effects else pd.DataFrame()
    )
    interaction_df = (
        pd.concat(interactions, ignore_index=True) if interactions else pd.DataFrame()
    )
    variance_df.to_csv(
        ctx.output_dir / "functional_anova_variance.csv", index=False
    )
    scene_df.to_csv(ctx.output_dir / "scene_effects.csv", index=False)
    camera_df.to_csv(ctx.output_dir / "camera_effects.csv", index=False)
    interaction_df.to_csv(
        ctx.output_dir / "scene_camera_interaction_effects.csv", index=False
    )
    return variance_df, scene_df, camera_df, interaction_df


def safe_correlation(
    x: Sequence[float], y: Sequence[float], method: str
) -> float:
    """Calculate a correlation safely when inputs have sufficient variation."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[valid], ya[valid]
    if len(xa) < 2 or np.ptp(xa) <= EPSILON or np.ptp(ya) <= EPSILON:
        return np.nan
    if method == "pearson":
        return float(stats.pearsonr(xa, ya).statistic)
    if method == "spearman":
        return float(stats.spearmanr(xa, ya).statistic)
    if method == "kendall":
        return float(stats.kendalltau(xa, ya, variant="b").statistic)
    raise ValueError(method)


def heldout_profiles(
    lap_df: pd.DataFrame,
    holdout_column: str = "scene_id",
    scene_id_column: str = "scene_id",
) -> pd.DataFrame:
    """Measure degradation camera-profile stability under leave-one-group-out."""
    data = target_laps(lap_df, "delta")
    value = TARGETS["delta"]
    rows: list[dict[str, Any]] = []
    levels = sorted(data[holdout_column].dropna().unique())
    if len(levels) < 2:
        return pd.DataFrame()
    for level in levels:
        train = data.loc[data[holdout_column] != level]
        heldout = data.loc[data[holdout_column] == level]
        train_means = train.groupby("camera_id", observed=True)[value].mean()
        heldout_means = heldout.groupby("camera_id", observed=True)[value].mean()
        common = train_means.index.intersection(heldout_means.index)
        train_common = train_means.loc[common]
        heldout_common = heldout_means.loc[common]
        if len(common):
            train_best = str(train_common.idxmin())
            heldout_best = str(heldout_common.idxmin())
            regret = float(
                heldout_common.loc[train_best] - heldout_common.min()
            )
        else:
            train_best, heldout_best, regret = "", "", np.nan
        rows.append(
            {
                "holdout_type": f"leave_one_{holdout_column}_out",
                "heldout_group": level,
                "scene_id": level if holdout_column == scene_id_column else "",
                "pearson": safe_correlation(
                    train_common.values, heldout_common.values, "pearson"
                ),
                "spearman": safe_correlation(
                    train_common.values, heldout_common.values, "spearman"
                ),
                "kendall_tau_b": safe_correlation(
                    train_common.values, heldout_common.values, "kendall"
                ),
                "train_best_camera": train_best,
                "heldout_best_camera": heldout_best,
                "top1_agreement": int(train_best == heldout_best)
                if len(common)
                else np.nan,
                "selection_regret": regret,
                "n_common_camera_settings": int(len(common)),
            }
        )
    return pd.DataFrame(rows)


def run_heldout_analysis(
    lap_df: pd.DataFrame, ctx: AnalysisContext
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-scene and optional leave-one-factor-level diagnostics."""
    tables = [heldout_profiles(lap_df, "scene_id")]
    for factor in ["topology", "light", "speed"]:
        if target_laps(lap_df, "delta")[factor].nunique() >= 2:
            tables.append(heldout_profiles(lap_df, factor))
    result = pd.concat(
        [table for table in tables if not table.empty],
        ignore_index=True,
    )
    result.to_csv(ctx.output_dir / "heldout_scene_correlations.csv", index=False)
    summary = (
        result.groupby("holdout_type", observed=True)
        .agg(
            n_holdouts=("heldout_group", "size"),
            mean_pearson=("pearson", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_kendall_tau_b=("kendall_tau_b", "mean"),
            top1_agreement_rate=("top1_agreement", "mean"),
            mean_selection_regret=("selection_regret", "mean"),
            median_selection_regret=("selection_regret", "median"),
            min_common_camera_settings=("n_common_camera_settings", "min"),
        )
        .reset_index()
    )
    summary.to_csv(ctx.output_dir / "heldout_scene_summary.csv", index=False)
    return result, summary


def effect_row(table: pd.DataFrame, effect: str) -> pd.Series:
    """Return an ANOVA effect row, or an empty NaN-valued series."""
    if table.empty or "effect" not in table:
        return pd.Series(dtype=float)
    rows = table.loc[table["effect"].eq(effect)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=float)


def scalar(row: pd.Series, column: str) -> float:
    """Extract a finite scalar from a row, otherwise NaN."""
    return float(row[column]) if column in row and pd.notna(row[column]) else np.nan


def suppression_metrics(
    variance_df: pd.DataFrame,
    anova_tables: dict[str, pd.DataFrame],
    ctx: AnalysisContext,
) -> pd.DataFrame:
    """Compare absolute raw and canonical-relative scene components."""
    rows: list[dict[str, Any]] = []
    for weighting in ["cell_balanced", "observed_weighted"]:
        raw_rows = variance_df.loc[
            variance_df["target"].eq("raw")
            & variance_df["weighting"].eq(weighting)
        ]
        delta_rows = variance_df.loc[
            variance_df["target"].eq("delta")
            & variance_df["weighting"].eq(weighting)
        ]
        if raw_rows.empty or delta_rows.empty:
            continue
        raw, delta = raw_rows.iloc[0], delta_rows.iloc[0]
        v_raw = float(raw["V_scene"])
        if abs(v_raw) <= EPSILON:
            reduction = np.nan
            ctx.warn(
                f"V_scene_raw is near zero for {weighting}; scene variance "
                "reduction ratio was not divided."
            )
        else:
            reduction = 1.0 - float(delta["V_scene"]) / v_raw
        raw_scene = effect_row(anova_tables["raw"], "scene")
        delta_scene = effect_row(anova_tables["delta"], "scene")
        rows.append(
            {
                "weighting": weighting,
                "V_scene_raw": v_raw,
                "V_scene_delta": float(delta["V_scene"]),
                "scene_variance_reduction_ratio": reduction,
                "fraction_scene_raw": float(raw["fraction_scene"]),
                "fraction_scene_delta": float(delta["fraction_scene"]),
                "scene_partial_eta_squared_raw": scalar(
                    raw_scene, "partial_eta_squared"
                ),
                "scene_partial_eta_squared_delta": scalar(
                    delta_scene, "partial_eta_squared"
                ),
                "camera_fraction_raw": float(raw["fraction_camera"]),
                "camera_fraction_delta": float(delta["fraction_camera"]),
                "scene_camera_fraction_raw": float(
                    raw["fraction_scene_camera"]
                ),
                "scene_camera_fraction_delta": float(
                    delta["fraction_scene_camera"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        ctx.output_dir / "scene_dependency_suppression.csv", index=False
    )
    return result


def bootstrap_statistics(
    lap_df: pd.DataFrame,
    point_variance: pd.DataFrame,
    point_suppression: pd.DataFrame,
    heldout_summary: pd.DataFrame,
    ctx: AnalysisContext,
) -> pd.DataFrame:
    """Cluster-bootstrap scenes and compute percentile confidence intervals."""
    output = ctx.output_dir / "bootstrap_confidence_intervals.csv"
    if ctx.args.skip_bootstrap:
        pd.DataFrame(
            columns=[
                "metric",
                "target",
                "weighting",
                "estimate",
                "ci_lower_95",
                "ci_upper_95",
                "n_successful_bootstrap",
            ]
        ).to_csv(output, index=False)
        ctx.logger.info("Bootstrap skipped by --skip-bootstrap")
        return pd.DataFrame()
    scenes = sorted(lap_df["scene_id"].dropna().unique())
    if len(scenes) < 5:
        ctx.warn(
            f"Only {len(scenes)} scenes are available; cluster-bootstrap confidence "
            "intervals may be unstable."
        )
    rng = np.random.default_rng(ctx.args.seed)
    samples: list[dict[str, Any]] = []
    for index in range(ctx.args.n_bootstrap):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        pieces: list[pd.DataFrame] = []
        for draw, scene in enumerate(chosen):
            piece = lap_df.loc[lap_df["scene_id"].eq(scene)].copy()
            piece["bootstrap_source_scene"] = scene
            piece["scene_id"] = f"{scene}__bootstrap_{draw}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            variances: dict[tuple[str, str], dict[str, float]] = {}
            for target in TARGETS:
                for weighting in ["cell_balanced", "observed_weighted"]:
                    variance, _, _, _ = functional_decomposition(
                        sample, target, weighting
                    )
                    variances[(target, weighting)] = variance
                    for metric in [
                        "fraction_scene",
                        "fraction_camera",
                        "fraction_scene_camera",
                        "fraction_residual",
                    ]:
                        samples.append(
                            {
                                "bootstrap": index,
                                "metric": metric,
                                "target": target,
                                "weighting": weighting,
                                "value": variance[metric],
                            }
                        )
            for weighting in ["cell_balanced", "observed_weighted"]:
                v_raw = variances[("raw", weighting)]["V_scene"]
                reduction = (
                    1 - variances[("delta", weighting)]["V_scene"] / v_raw
                    if abs(v_raw) > EPSILON
                    else np.nan
                )
                samples.append(
                    {
                        "bootstrap": index,
                        "metric": "scene_variance_reduction_ratio",
                        "target": "raw_vs_delta",
                        "weighting": weighting,
                        "value": reduction,
                    }
                )
            held = heldout_profiles(sample, "bootstrap_source_scene")
            for metric, column in [
                ("mean_heldout_pearson", "pearson"),
                ("mean_heldout_spearman", "spearman"),
                ("mean_heldout_kendall_tau_b", "kendall_tau_b"),
                ("mean_selection_regret", "selection_regret"),
            ]:
                samples.append(
                    {
                        "bootstrap": index,
                        "metric": metric,
                        "target": "delta",
                        "weighting": "cluster_scene",
                        "value": held[column].mean(),
                    }
                )
        except Exception as exc:
            ctx.warn(f"Bootstrap replicate {index} failed: {exc}")
        if (index + 1) % max(1, ctx.args.n_bootstrap // 10) == 0:
            ctx.logger.info(
                "Bootstrap progress: %d/%d", index + 1, ctx.args.n_bootstrap
            )
    sample_df = pd.DataFrame(samples)
    rows: list[dict[str, Any]] = []
    for (metric, target, weighting), group in sample_df.groupby(
        ["metric", "target", "weighting"], observed=True
    ):
        values = group["value"].dropna()
        estimate = np.nan
        if metric.startswith("fraction_"):
            point = point_variance.loc[
                point_variance["target"].eq(target)
                & point_variance["weighting"].eq(weighting)
            ]
            if not point.empty:
                estimate = float(point.iloc[0][metric])
        elif metric == "scene_variance_reduction_ratio":
            point = point_suppression.loc[
                point_suppression["weighting"].eq(weighting)
            ]
            if not point.empty:
                estimate = float(
                    point.iloc[0]["scene_variance_reduction_ratio"]
                )
        else:
            scene_summary = heldout_summary.loc[
                heldout_summary["holdout_type"].eq("leave_one_scene_id_out")
            ]
            mapping = {
                "mean_heldout_pearson": "mean_pearson",
                "mean_heldout_spearman": "mean_spearman",
                "mean_heldout_kendall_tau_b": "mean_kendall_tau_b",
                "mean_selection_regret": "mean_selection_regret",
            }
            if not scene_summary.empty:
                estimate = float(scene_summary.iloc[0][mapping[metric]])
        rows.append(
            {
                "metric": metric,
                "target": target,
                "weighting": weighting,
                "estimate": estimate,
                "ci_lower_95": values.quantile(0.025) if len(values) else np.nan,
                "ci_upper_95": values.quantile(0.975) if len(values) else np.nan,
                "n_successful_bootstrap": int(len(values)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


def _matrix(
    data: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Build a consistently sorted pivot table."""
    return data.pivot_table(
        index=index, columns=columns, values=values, aggfunc=aggfunc
    ).sort_index().sort_index(axis=1)


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    cmap: str = "viridis",
    annotate: bool = False,
    centered: bool = False,
) -> None:
    """Save a labeled heatmap with optional cell annotations."""
    width = max(7.0, min(22.0, 0.55 * max(1, matrix.shape[1]) + 4))
    height = max(5.0, min(20.0, 0.38 * max(1, matrix.shape[0]) + 3))
    fig, ax = plt.subplots(figsize=(width, height))
    values = matrix.to_numpy(dtype=float)
    kwargs: dict[str, Any] = {"aspect": "auto", "cmap": cmap}
    if centered and np.isfinite(values).any():
        bound = float(np.nanmax(np.abs(values)))
        kwargs.update(vmin=-bound, vmax=bound)
    image = ax.imshow(values, **kwargs)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=60, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    if annotate and matrix.size <= 100:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = values[row, column]
                if np.isfinite(value):
                    ax.text(
                        column,
                        row,
                        f"{value:.3g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(
    data: pd.DataFrame,
    group: str,
    path: Path,
    title: str,
) -> None:
    """Plot mean degradation camera profiles by a scene descriptor."""
    value = TARGETS["delta"]
    profile = (
        data.groupby([group, "camera_id"], observed=True)[value]
        .mean()
        .unstack("camera_id")
    )
    fig, ax = plt.subplots(figsize=(max(10, profile.shape[1] * 0.65), 6))
    x = np.arange(profile.shape[1])
    for label, row in profile.iterrows():
        ax.plot(x, row.values, marker="o", linewidth=1.2, label=str(label))
    ax.set_xticks(x)
    ax.set_xticklabels(profile.columns.astype(str), rotation=60, ha="right")
    ax.set_xlabel("Camera setting (exposure × gain)")
    ax.set_ylabel("Mean canonical-relative AbsRel degradation")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=max(1, min(4, profile.shape[0] // 5 + 1)))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    frame_df: pd.DataFrame,
    lap_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    heldout: pd.DataFrame,
    match_rates: pd.DataFrame,
    ctx: AnalysisContext,
) -> None:
    """Generate all required headless PNG figures."""
    figure_dir = ctx.output_dir / "figures"
    delta = target_laps(lap_df, "delta")
    balance = (
        lap_df.groupby(["scene_id", "camera_id"], observed=True)
        .size()
        .reset_index(name="n_laps")
    )
    save_heatmap(
        _matrix(balance, "scene_id", "camera_id", "n_laps"),
        figure_dir / "01_dataset_balance_heatmap.png",
        "Lap balance by scene and camera",
        "Camera setting",
        "Scene",
        "Number of laps",
        cmap="Blues",
        annotate=True,
    )
    overall = (
        delta.groupby(["source_exposure", "source_gain"], observed=True)
        [TARGETS["delta"]]
        .mean()
        .reset_index()
    )
    save_heatmap(
        _matrix(
            overall,
            "source_exposure",
            "source_gain",
            TARGETS["delta"],
        ),
        figure_dir / "02_overall_camera_degradation_heatmap.png",
        "Overall camera degradation (lap-level mean)",
        "Source gain",
        "Source exposure",
        "Mean degradation",
        annotate=True,
    )
    save_heatmap(
        _matrix(delta, "scene_id", "camera_id", TARGETS["delta"]),
        figure_dir / "03_scene_camera_degradation_heatmap.png",
        "Canonical-relative degradation by scene and camera",
        "Camera setting",
        "Scene",
        "Mean degradation",
    )
    interaction = interaction_df.loc[
        interaction_df["target"].eq("delta")
        & interaction_df["weighting"].eq("cell_balanced")
    ]
    save_heatmap(
        _matrix(
            interaction,
            "scene_id",
            "camera_id",
            "f_scene_camera",
        ),
        figure_dir / "04_scene_camera_interaction_heatmap.png",
        "Scene × camera functional interaction",
        "Camera setting",
        "Scene",
        "Interaction effect",
        cmap="coolwarm",
        centered=True,
    )
    selected = variance_df.loc[
        variance_df["weighting"].eq("cell_balanced")
    ].set_index("target")
    components = [
        "fraction_scene",
        "fraction_camera",
        "fraction_scene_camera",
        "fraction_residual",
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(components))
    width = 0.36
    for offset, target in [(-width / 2, "raw"), (width / 2, "delta")]:
        values = (
            selected.loc[target, components].astype(float).values
            if target in selected.index
            else np.full(len(components), np.nan)
        )
        ax.bar(positions + offset, values, width=width, label=target)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Scene", "Camera", "Scene×camera", "Residual"])
    ax.set_ylabel("Variance fraction")
    ax.set_title("Functional variance decomposition: raw vs delta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figure_dir / "05_variance_decomposition_raw_vs_delta.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    plot_profiles(
        delta,
        "scene_id",
        figure_dir / "06_camera_profile_by_scene.png",
        "Camera degradation profile by scene",
    )
    plot_profiles(
        delta,
        "topology",
        figure_dir / "07_camera_profile_by_topology.png",
        "Camera degradation profile by topology",
    )
    plot_profiles(
        delta,
        "light",
        figure_dir / "08_camera_profile_by_light.png",
        "Camera degradation profile by illumination",
    )
    plot_profiles(
        delta,
        "speed",
        figure_dir / "09_camera_profile_by_speed.png",
        "Camera degradation profile by speed",
    )

    scene_held = heldout.loc[
        heldout["holdout_type"].eq("leave_one_scene_id_out")
    ]
    fig, ax = plt.subplots(
        figsize=(max(10, len(scene_held) * 0.5), 5.5)
    )
    x = np.arange(len(scene_held))
    ax.plot(x, scene_held["pearson"], marker="o", label="Pearson")
    ax.plot(x, scene_held["spearman"], marker="s", label="Spearman")
    ax.plot(x, scene_held["kendall_tau_b"], marker="^", label="Kendall tau-b")
    ax.set_xticks(x)
    ax.set_xticklabels(scene_held["heldout_group"], rotation=60, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Held-out scene")
    ax.set_ylabel("Correlation")
    ax.set_title("Leave-one-scene-out camera-profile stability")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "10_heldout_scene_correlations.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    match_cell = match_rates.loc[match_rates["grouping"].eq("scene_camera")]
    save_heatmap(
        _matrix(match_cell, "scene_id", "camera_id", "match_rate"),
        figure_dir / "11_matching_rate_heatmap.png",
        "Registration match rate by scene and camera",
        "Camera setting",
        "Scene",
        "Matched fraction",
        cmap="magma",
    )


def format_number(value: Any, digits: int = 4) -> str:
    """Format report numbers safely, using scientific notation when helpful."""
    number = finite_float(value)
    if number is None:
        return "NA"
    if number != 0 and (abs(number) < 10 ** (-digits) or abs(number) >= 1e4):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def describe_anova_effect(table: pd.DataFrame, effect: str) -> str:
    """Create a compact F/p/effect-size result string."""
    row = effect_row(table, effect)
    residual = effect_row(table, "residual")
    if row.empty:
        return "not estimable"
    return (
        f"F({format_number(row.get('df'), 1)}, "
        f"{format_number(residual.get('df'), 1)})="
        f"{format_number(row.get('F'))}, p={format_number(row.get('p_value'))}, "
        f"partial eta squared={format_number(row.get('partial_eta_squared'))}"
    )


def paper_f_stat(table: pd.DataFrame, effect: str) -> str:
    """Format an ANOVA statistic in paper-style F(df1, df2) notation."""
    row = effect_row(table, effect)
    residual = effect_row(table, "residual")
    if row.empty:
        return "not estimable"
    return (
        f"F({format_number(row.get('df'), 1)}, "
        f"{format_number(residual.get('df'), 1)})="
        f"{format_number(row.get('F'))}, "
        f"p={format_number(row.get('p_value'))}, partial eta squared="
        f"{format_number(row.get('partial_eta_squared'))}"
    )


def build_results_summary(
    quality: dict[str, Any],
    lap_df: pd.DataFrame,
    anova_tables: dict[str, pd.DataFrame],
    fits: dict[str, dict[str, float]],
    variance_df: pd.DataFrame,
    suppression: pd.DataFrame,
    heldout_summary: pd.DataFrame,
    ctx: AnalysisContext,
) -> pd.DataFrame:
    """Build one compact summary row per target and weighting scheme."""
    scene_held = heldout_summary.loc[
        heldout_summary["holdout_type"].eq("leave_one_scene_id_out")
    ]
    held = scene_held.iloc[0] if not scene_held.empty else pd.Series(dtype=float)
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for weighting in ["cell_balanced", "observed_weighted"]:
            variance_rows = variance_df.loc[
                variance_df["target"].eq(target)
                & variance_df["weighting"].eq(weighting)
            ]
            variance = (
                variance_rows.iloc[0] if not variance_rows.empty else pd.Series()
            )
            sup_rows = suppression.loc[suppression["weighting"].eq(weighting)]
            sup = sup_rows.iloc[0] if not sup_rows.empty else pd.Series()
            scene = effect_row(anova_tables[target], "scene")
            camera = effect_row(anova_tables[target], "camera")
            interaction = effect_row(anova_tables[target], "scene_camera")
            rows.append(
                {
                    "target": target,
                    "weighting": weighting,
                    "n_files": quality["n_csv_included"],
                    "n_frames": quality["n_frames"],
                    "n_scenes": int(
                        target_laps(lap_df, target)["scene_id"].nunique()
                    ),
                    "n_topologies": int(
                        target_laps(lap_df, target)["topology"].nunique()
                    ),
                    "n_lights": int(
                        target_laps(lap_df, target)["light"].nunique()
                    ),
                    "n_speeds": int(
                        target_laps(lap_df, target)["speed"].nunique()
                    ),
                    "n_cameras": quality["n_camera_settings"],
                    "n_laps": int(len(target_laps(lap_df, target))),
                    "r_squared": fits[target]["r_squared"],
                    "adjusted_r_squared": fits[target]["adjusted_r_squared"],
                    "scene_F": scalar(scene, "F"),
                    "scene_p": scalar(scene, "p_value"),
                    "scene_eta_squared": scalar(scene, "eta_squared"),
                    "scene_partial_eta_squared": scalar(
                        scene, "partial_eta_squared"
                    ),
                    "camera_F": scalar(camera, "F"),
                    "camera_p": scalar(camera, "p_value"),
                    "camera_eta_squared": scalar(camera, "eta_squared"),
                    "camera_partial_eta_squared": scalar(
                        camera, "partial_eta_squared"
                    ),
                    "scene_camera_F": scalar(interaction, "F"),
                    "scene_camera_p": scalar(interaction, "p_value"),
                    "scene_camera_eta_squared": scalar(
                        interaction, "eta_squared"
                    ),
                    "scene_camera_partial_eta_squared": scalar(
                        interaction, "partial_eta_squared"
                    ),
                    "fraction_scene": variance.get("fraction_scene", np.nan),
                    "fraction_camera": variance.get("fraction_camera", np.nan),
                    "fraction_scene_camera": variance.get(
                        "fraction_scene_camera", np.nan
                    ),
                    "fraction_residual": variance.get(
                        "fraction_residual", np.nan
                    ),
                    "scene_variance_reduction_ratio": sup.get(
                        "scene_variance_reduction_ratio", np.nan
                    ),
                    "mean_heldout_pearson": held.get(
                        "mean_pearson", np.nan
                    ),
                    "mean_heldout_spearman": held.get(
                        "mean_spearman", np.nan
                    ),
                    "mean_heldout_kendall": held.get(
                        "mean_kendall_tau_b", np.nan
                    ),
                    "mean_selection_regret": held.get(
                        "mean_selection_regret", np.nan
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(ctx.output_dir / "results_summary.csv", index=False)
    return result


def decomposed_text(table: pd.DataFrame, effect: str) -> str:
    """Format one decomposed ANOVA interaction result."""
    row = effect_row(table, effect)
    if row.empty:
        return "not estimable (factor/term was absent or aliased)"
    return (
        f"F={format_number(row.get('F'))}, p={format_number(row.get('p_value'))}, "
        f"partial eta squared={format_number(row.get('partial_eta_squared'))}"
    )


def report_documents(
    quality: dict[str, Any],
    anova_tables: dict[str, pd.DataFrame],
    decomposed: dict[str, pd.DataFrame],
    variance_df: pd.DataFrame,
    suppression: pd.DataFrame,
    heldout_summary: pd.DataFrame,
    match_rates: pd.DataFrame,
    ctx: AnalysisContext,
) -> None:
    """Generate Markdown and plain-text reports using computed results."""
    delta_camera = effect_row(anova_tables["delta"], "camera")
    delta_interaction = effect_row(anova_tables["delta"], "scene_camera")
    raw_camera = effect_row(anova_tables["raw"], "camera")
    raw_interaction = effect_row(anova_tables["raw"], "scene_camera")
    delta_scene = effect_row(anova_tables["delta"], "scene")
    balanced_var = variance_df.loc[
        variance_df["target"].eq("delta")
        & variance_df["weighting"].eq("cell_balanced")
    ]
    delta_var = balanced_var.iloc[0] if not balanced_var.empty else pd.Series()
    sup_rows = suppression.loc[
        suppression["weighting"].eq("cell_balanced")
    ]
    sup = sup_rows.iloc[0] if not sup_rows.empty else pd.Series()
    held_rows = heldout_summary.loc[
        heldout_summary["holdout_type"].eq("leave_one_scene_id_out")
    ]
    held = held_rows.iloc[0] if not held_rows.empty else pd.Series()
    match_cell = match_rates.loc[match_rates["grouping"].eq("scene_camera")]
    match_min = match_cell["match_rate"].min()
    match_max = match_cell["match_rate"].max()
    camera_sig = scalar(delta_camera, "p_value") < 0.05
    interaction_sig = scalar(delta_interaction, "p_value") < 0.05
    reduction = sup.get("scene_variance_reduction_ratio", np.nan)
    reduction_pct = 100 * reduction if pd.notna(reduction) else np.nan
    camera_fraction_pct = 100 * delta_var.get("fraction_camera", np.nan)
    interaction_fraction_pct = 100 * delta_var.get(
        "fraction_scene_camera", np.nan
    )
    scene_fraction_pct = 100 * delta_var.get("fraction_scene", np.nan)

    if camera_sig:
        global_camera = (
            "A statistically detectable global camera-induced degradation pattern "
            "exists across the observed scenes."
        )
    else:
        global_camera = (
            "No statistically detectable global camera-induced degradation pattern "
            "was established at alpha=0.05."
        )
    if interaction_sig:
        interaction_interpretation = (
            "The effect of camera parameters varies across scenes, indicating that "
            "a camera-only lookup table is insufficient and supporting an "
            "image-conditioned scalar-risk model. The camera main effect represents "
            "an average tendency across scenes, not a scene-invariant deterministic "
            "degradation value."
        )
    else:
        interaction_interpretation = (
            "The scene-camera interaction was not statistically detectable at "
            "alpha=0.05; this does not establish exact scene invariance."
        )
    if pd.notna(reduction) and reduction > 0:
        suppression_text = (
            "Canonical-relative degradation reduced the estimated absolute scene "
            f"component by {reduction_pct:.2f}%."
        )
    else:
        suppression_text = (
            "Canonical-relative degradation did not show a positive reduction of "
            "the estimated absolute scene component under this decomposition."
        )
    if pd.notna(scene_fraction_pct) and scene_fraction_pct > 10:
        suppression_text += " Scene dependency was reduced but not eliminated."

    english = (
        "Across the evaluated topology, illumination, and motion conditions, "
        f"camera parameters exhibited a {'significant' if camera_sig else 'non-significant'} "
        "global effect on canonical-relative depth degradation "
        f"({paper_f_stat(anova_tables['delta'], 'camera')}). "
        f"The scene-camera interaction was "
        f"{'significant' if interaction_sig else 'non-significant'} "
        f"({paper_f_stat(anova_tables['delta'], 'scene_camera')}), "
        f"{'indicating scene-dependent camera sensitivity' if interaction_sig else 'providing no alpha=0.05 evidence of scene-dependent camera sensitivity'}. "
        f"Relative to raw AbsRel, canonical-relative degradation changed the "
        f"estimated scene component by a reduction of {format_number(reduction_pct, 2)}%, "
        f"while camera and scene-camera effects accounted for "
        f"{format_number(camera_fraction_pct, 2)}% and "
        f"{format_number(interaction_fraction_pct, 2)}% of the component-sum "
        "variance, respectively."
    )
    korean = (
        "평가된 topology, 조명, 움직임 조건에서 camera parameter의 "
        f"canonical-relative depth degradation에 대한 전역 효과는 "
        f"{'통계적으로 유의하였다' if camera_sig else '통계적으로 유의하지 않았다'} "
        f"({paper_f_stat(anova_tables['delta'], 'camera')}). "
        f"scene-camera interaction은 "
        f"{'유의하였다' if interaction_sig else '유의하지 않았다'} "
        f"({paper_f_stat(anova_tables['delta'], 'scene_camera')}). "
        f"Raw AbsRel 대비 canonical-relative degradation의 추정 scene 성분 감소율은 "
        f"{format_number(reduction_pct, 2)}%였으며, camera와 scene-camera 성분은 "
        f"각각 component-sum variance의 {format_number(camera_fraction_pct, 2)}%와 "
        f"{format_number(interaction_fraction_pct, 2)}%를 설명하였다."
    )
    warnings_md = (
        "\n".join(f"- {warning}" for warning in ctx.warnings)
        if ctx.warnings
        else "- No pipeline warning was recorded."
    )
    report = f"""# Camera–Scene Functional ANOVA Report

## 1. Analysis Scope

This analysis decomposes lap-level image depth error into scene, camera, scene-camera interaction, and residual components. Scene is topology × light × the CSV-internal `source_motion_label`; camera is the categorical exposure × gain pair. Input CSV files were never modified.

## 2. Dataset Summary

- CSV files: {quality['n_csv_included']} included / {quality['n_csv_discovered']} discovered
- Frames: {quality['n_frames']}; matched: {quality['n_matched_rows']}; registration_failed: {quality['n_registration_failed_rows']}
- Scenes: {quality['n_scenes']}; topologies: {quality['n_topologies']}; lights: {quality['n_lights']}; motion labels: {quality['n_speeds']}
- Camera settings: {quality['n_camera_settings']}; retained lap units: {quality['n_laps_retained_any_target']}
- Delta model scenes/motion labels: {quality['n_delta_scenes']}/{', '.join(quality['delta_motion_levels'])}

## 3. Data Quality and Missing Cells

The design was **{'balanced' if quality['balanced'] else 'unbalanced'}**. Missing scene-camera cells: {quality['n_missing_scene_camera_cells']}. Cell lap counts ranged from {quality['scene_camera_lap_count_min']} to {quality['scene_camera_lap_count_max']}. Possible duplicated analysis-key rows: {quality['possible_duplicate_rows']}. Detailed counts are in `scene_camera_balance.csv`, `missing_scene_camera_cells.csv`, and `data_quality_summary.json`.

## 4. Statistical Models

Primary OLS models used lap means, Sum contrasts, and Type III sums of squares. P-values are interpreted together with eta squared, partial eta squared, omega squared, and functional variance fractions. Frame rows were not treated as independent replicates.

## 5. Scene × Camera ANOVA: Canonical-Relative Degradation

- Scene: {describe_anova_effect(anova_tables['delta'], 'scene')}
- Camera: {describe_anova_effect(anova_tables['delta'], 'camera')}
- Scene × camera: {describe_anova_effect(anova_tables['delta'], 'scene_camera')}

{global_camera} {interaction_interpretation}

## 6. Scene × Camera ANOVA: Raw AbsRel

- Scene: {describe_anova_effect(anova_tables['raw'], 'scene')}
- Camera: {describe_anova_effect(anova_tables['raw'], 'camera')}
- Scene × camera: {describe_anova_effect(anova_tables['raw'], 'scene_camera')}

## 7. Functional Variance Decomposition

For canonical-relative degradation under cell-balanced weighting: scene={format_number(delta_var.get('fraction_scene'))}, camera={format_number(delta_var.get('fraction_camera'))}, scene-camera={format_number(delta_var.get('fraction_scene_camera'))}, residual={format_number(delta_var.get('fraction_residual'))}. Both cell-balanced and observed-weighted estimates are in `functional_anova_variance.csv`.

## 8. Scene Dependency Suppression

{suppression_text} Raw and delta targets can have different numerical scales, so the absolute variance ratio is a descriptive comparison and not a scale-free causal quantity.

## 9. Global Camera Effect

{global_camera} Where interaction is appreciable, this is an average tendency across observed scenes rather than a universal deterministic lookup value.

## 10. Scene × Camera Interaction

{interaction_interpretation}

## 11. Topology × Camera

- Delta: {decomposed_text(decomposed['delta'], 'topology_camera')}
- Raw: {decomposed_text(decomposed['raw'], 'topology_camera')}

This term diagnoses whether object arrangement or scene composition changes camera-parameter sensitivity.

## 12. Light × Camera

- Delta: {decomposed_text(decomposed['delta'], 'light_camera')}
- Raw: {decomposed_text(decomposed['raw'], 'light_camera')}

This term diagnoses whether illumination changes exposure/gain sensitivity.

## 13. Speed × Camera

- Delta: {decomposed_text(decomposed['delta'], 'speed_camera')}
- Raw: {decomposed_text(decomposed['raw'], 'speed_camera')}

This term diagnoses motion-blur and exposure sensitivity across the CSV-internal motion labels (`source_motion_label`).

## 14. Held-Out Scene Stability

Mean leave-one-scene-out correlations were Pearson={format_number(held.get('mean_pearson'))}, Spearman={format_number(held.get('mean_spearman'))}, and Kendall tau-b={format_number(held.get('mean_kendall_tau_b'))}; mean camera-selection regret was {format_number(held.get('mean_selection_regret'), 6)}. The average camera degradation profile was relatively stable across observed scene groups only when these correlations are high. This does not prove unrestricted generalization to arbitrary unseen environments.

## 15. Registration Selection Bias

Scene-camera match rates ranged from {format_number(match_min)} to {format_number(match_max)}. Match-rate heterogeneity means difficult frames may be selectively excluded by matched-only analysis, potentially biasing degradation downward. Inspect `registration_match_rate_analysis.csv`, `11_matching_rate_heatmap.png`, and `registration_glm_summary.txt`.

## 16. Main Findings

1. {global_camera}
2. {interaction_interpretation}
3. {suppression_text}
4. Held-out profile correlations and selection regret quantify stability only within the observed factor levels.

## 17. Limitations

- Observational associations do not establish causal camera effects.
- Different raw and delta scales limit direct absolute-variance comparisons.
- Missing/unequal cells, registration failures, and matched-only filtering can bias estimates.
- Cluster-bootstrap inference is limited by the number and diversity of scenes.
- Held-out tests do not prove generalization to arbitrary unseen environments.

Recorded warnings:

{warnings_md}

## 18. Paper-Ready Interpretation

**English.** {english}

**한국어.** {korean}
"""
    (ctx.output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    plain = re.sub(r"[#*_`]", "", report)
    (ctx.output_dir / "analysis_report.txt").write_text(plain, encoding="utf-8")


def registration_bias_warnings(
    match_rates: pd.DataFrame, ctx: AnalysisContext
) -> None:
    """Record explicit matched-only selection-bias warnings."""
    scene_camera = match_rates.loc[
        match_rates["grouping"].eq("scene_camera")
    ]
    camera = match_rates.loc[match_rates["grouping"].eq("camera")]
    cell_range = (
        float(scene_camera["match_rate"].max() - scene_camera["match_rate"].min())
        if not scene_camera.empty
        else np.nan
    )
    camera_range = (
        float(camera["match_rate"].max() - camera["match_rate"].min())
        if not camera.empty
        else np.nan
    )
    if np.isfinite(cell_range) and cell_range >= 0.10:
        ctx.warn(
            f"Scene-camera registration match rates differ substantially "
            f"(range={cell_range:.3f})."
        )
    if np.isfinite(camera_range) and camera_range >= 0.05:
        ctx.warn(
            f"Registration failures are concentrated unevenly across camera "
            f"settings (camera-level rate range={camera_range:.3f})."
        )
    ctx.warn(
        "Matched-only analysis may selectively exclude difficult frames; estimated "
        "degradation may therefore be biased downward."
    )


def current_git_commit() -> str | None:
    """Return the current Git commit when the repository is available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def write_run_metadata(
    ctx: AnalysisContext, files_metadata: list[dict[str, Any]]
) -> None:
    """Write reproducibility metadata including versions and file hashes."""
    payload = {
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "cli_arguments": vars(ctx.args),
        "random_seed": ctx.args.seed,
        "searched_csv_files": files_metadata,
        "git_commit": current_git_commit(),
    }
    write_json(ctx.output_dir / "run_metadata.json", payload)


def terminal_summary(
    quality: dict[str, Any],
    anova_tables: dict[str, pd.DataFrame],
    suppression: pd.DataFrame,
    heldout_summary: pd.DataFrame,
    match_rates: pd.DataFrame,
    ctx: AnalysisContext,
) -> None:
    """Print the required compact terminal summary."""
    camera = effect_row(anova_tables["delta"], "camera")
    interaction = effect_row(anova_tables["delta"], "scene_camera")
    sup_rows = suppression.loc[
        suppression["weighting"].eq("cell_balanced")
    ]
    sup = sup_rows.iloc[0] if not sup_rows.empty else pd.Series()
    held_rows = heldout_summary.loc[
        heldout_summary["holdout_type"].eq("leave_one_scene_id_out")
    ]
    held = held_rows.iloc[0] if not held_rows.empty else pd.Series()
    rates = match_rates.loc[
        match_rates["grouping"].eq("scene_camera"), "match_rate"
    ]
    report_path = (ctx.output_dir / "analysis_report.md").resolve()
    print("\n=== Camera–Scene fANOVA Summary ===")
    print(f"Discovered CSV files: {quality['n_csv_discovered']}")
    print(
        "Scenes/topologies/lights/speeds: "
        f"{quality['n_delta_scenes']}/{quality['n_topologies']}/"
        f"{quality['n_lights']}/{len(quality['delta_motion_levels'])}"
    )
    print(f"Camera settings: {quality['n_camera_settings']}")
    print(f"Retained lap units: {quality['n_laps_retained_any_target']}")
    print(
        "Camera effect: "
        f"F={format_number(camera.get('F'))}, "
        f"p={format_number(camera.get('p_value'))}, "
        f"partial eta^2={format_number(camera.get('partial_eta_squared'))}"
    )
    print(
        "Scene×camera effect: "
        f"F={format_number(interaction.get('F'))}, "
        f"p={format_number(interaction.get('p_value'))}, "
        f"partial eta^2={format_number(interaction.get('partial_eta_squared'))}"
    )
    print(
        "Scene variance reduction ratio: "
        f"{format_number(sup.get('scene_variance_reduction_ratio'))}"
    )
    print(
        "Held-out means (Pearson/Spearman/Kendall): "
        f"{format_number(held.get('mean_pearson'))}/"
        f"{format_number(held.get('mean_spearman'))}/"
        f"{format_number(held.get('mean_kendall_tau_b'))}"
    )
    print(
        "Registration match-rate range: "
        f"{format_number(rates.min())}–{format_number(rates.max())}"
    )
    print(f"Report: {report_path}")


def main() -> int:
    """Run the complete camera–scene functional ANOVA pipeline."""
    args = parse_args()
    ctx = setup_context(args)
    try:
        frame_df, excluded_files, inventory, files_metadata = discover_and_load(ctx)
        frame_df = add_validity_flags(frame_df, ctx)
        lap_df, _ = aggregate_laps(frame_df, ctx)
        match_rates = registration_match_rates(frame_df, ctx)
        registration_bias_warnings(match_rates, ctx)
        quality = dataset_quality(
            frame_df,
            lap_df,
            excluded_files,
            inventory,
            match_rates,
            ctx,
        )
        anova_tables: dict[str, pd.DataFrame] = {}
        fits: dict[str, dict[str, float]] = {}
        decomposed: dict[str, pd.DataFrame] = {}
        for target in TARGETS:
            table, _, fit = fit_primary_anova(lap_df, target, ctx)
            anova_tables[target] = table
            fits[target] = fit
            decomposed[target], _ = fit_decomposed_anova(
                lap_df, target, ctx
            )
        variance, _, _, interactions = run_functional_anova(lap_df, ctx)
        suppression = suppression_metrics(variance, anova_tables, ctx)
        heldout, heldout_summary = run_heldout_analysis(lap_df, ctx)
        bootstrap_statistics(
            lap_df, variance, suppression, heldout_summary, ctx
        )
        build_results_summary(
            quality,
            lap_df,
            anova_tables,
            fits,
            variance,
            suppression,
            heldout_summary,
            ctx,
        )
        make_figures(
            frame_df,
            lap_df,
            variance,
            interactions,
            heldout,
            match_rates,
            ctx,
        )
        report_documents(
            quality,
            anova_tables,
            decomposed,
            variance,
            suppression,
            heldout_summary,
            match_rates,
            ctx,
        )
        write_run_metadata(ctx, files_metadata)
        terminal_summary(
            quality,
            anova_tables,
            suppression,
            heldout_summary,
            match_rates,
            ctx,
        )
        return 0
    except Exception:
        ctx.logger.exception("Fatal pipeline error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
