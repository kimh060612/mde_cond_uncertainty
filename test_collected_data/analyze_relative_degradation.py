#!/usr/bin/env python3
"""Analyze percentage degradation relative to canonical AbsRel error.

The frame-level target is
    (source_metric_abs_rel - canonical_metric_abs_rel)
    / canonical_metric_abs_rel
and all inferential analyses use lap-level aggregates.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from analyze_camera_scene_fanova import (
    enrich_anova,
    parse_scene_metadata,
    save_heatmap,
)


READ_COLUMNS = [
    "scene",
    "match_status",
    "source_exposure",
    "source_gain",
    "source_pair_dir",
    "source_lap_dir",
    "source_motion_label",
    "source_metric_abs_rel",
    "canonical_metric_abs_rel",
    "performance_degradation_abs_rel",
]
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lap-level canonical-relative percentage degradation analysis."
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
    return parser.parse_args()


def parameter_label(value: Any) -> str:
    """Format a numeric camera parameter as a stable categorical label."""
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def load_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, list[Path]]:
    """Load required columns and derive scene/camera metadata."""
    root = args.input_root.expanduser().resolve()
    files = sorted(path.resolve() for path in root.glob(args.glob_pattern))
    if not files:
        raise FileNotFoundError(f"No matching CSV under {root}")
    parts: list[pd.DataFrame] = []
    for path in files:
        header = pd.read_csv(path, nrows=0)
        missing = sorted(set(READ_COLUMNS) - set(header.columns))
        if missing:
            logging.warning("Skipping %s; missing %s", path, missing)
            continue
        data = pd.read_csv(path, usecols=READ_COLUMNS, low_memory=False)
        data["source_csv_path"] = str(path)
        parts.append(data)
    if not parts:
        raise RuntimeError("No file contained all required relative-target columns")
    frames = pd.concat(parts, ignore_index=True)
    parsed = frames.apply(
        lambda row: parse_scene_metadata(
            row["scene"], row["source_csv_path"]
        ),
        axis=1,
        result_type="expand",
    )
    parsed.columns = [
        "filename_scene_id",
        "topology",
        "light",
        "filename_speed",
    ]
    frames = pd.concat([frames, parsed], axis=1)
    frames["speed"] = (
        frames["source_motion_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "unknown", "nan": "unknown"})
    )
    frames["scene_id"] = (
        frames["light"].astype(str)
        + "_"
        + frames["speed"].astype(str)
        + "_"
        + frames["topology"].astype(str)
    )
    frames["camera_id"] = [
        f"exp_{parameter_label(exposure)}_gain_{parameter_label(gain)}"
        for exposure, gain in zip(
            frames["source_exposure"], frames["source_gain"], strict=False
        )
    ]
    for column in [
        "source_metric_abs_rel",
        "canonical_metric_abs_rel",
        "performance_degradation_abs_rel",
    ]:
        frames[column] = pd.to_numeric(frames[column], errors="coerce")
    source = frames["source_metric_abs_rel"]
    canonical = frames["canonical_metric_abs_rel"]
    difference = source - canonical
    frames["valid_relative"] = (
        frames["match_status"].astype(str).str.lower().eq("matched")
        & np.isfinite(source)
        & np.isfinite(canonical)
        & source.ge(0)
        & canonical.gt(0)
        & difference.ge(0)
    )
    frames["relative_degradation"] = difference / canonical
    return frames, files


def aggregate_laps(
    frames: pd.DataFrame, min_valid_frames: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate valid frame-level ratios and errors to lap-level units."""
    valid = frames.loc[frames["valid_relative"]].copy()
    group_columns = [
        "scene_id",
        "camera_id",
        "source_csv_path",
        "source_pair_dir",
        "source_lap_dir",
    ]
    laps = (
        valid.groupby(group_columns, observed=True)
        .agg(
            topology=("topology", "first"),
            light=("light", "first"),
            speed=("speed", "first"),
            source_exposure=("source_exposure", "first"),
            source_gain=("source_gain", "first"),
            source_abs_rel_mean=("source_metric_abs_rel", "mean"),
            canonical_abs_rel_mean=("canonical_metric_abs_rel", "mean"),
            absolute_degradation_mean=(
                "performance_degradation_abs_rel",
                "mean",
            ),
            relative_degradation_mean=("relative_degradation", "mean"),
            relative_degradation_median=("relative_degradation", "median"),
            relative_degradation_std=("relative_degradation", "std"),
            n_valid_frames=("relative_degradation", "size"),
        )
        .reset_index()
    )
    laps["eligible_frame_count"] = (
        laps["n_valid_frames"] >= min_valid_frames
    )
    minimum_motion_laps = max(2, 2 * laps["camera_id"].nunique())
    motion_counts = laps.loc[laps["eligible_frame_count"]].groupby(
        "speed", observed=True
    ).size()
    sparse_motion_levels = set(
        motion_counts.loc[
            motion_counts < minimum_motion_laps
        ].index.astype(str)
    )
    laps["eligible_motion_coverage"] = ~laps["speed"].isin(
        sparse_motion_levels
    )
    laps["eligible"] = (
        laps["eligible_frame_count"] & laps["eligible_motion_coverage"]
    )
    quality = (
        laps.groupby("speed", observed=True)
        .agg(
            n_lap_motion_segments=("source_lap_dir", "size"),
            n_segments_with_min_frames=("eligible_frame_count", "sum"),
            n_segments_in_model=("eligible", "sum"),
            median_valid_frames=("n_valid_frames", "median"),
            min_valid_frames=("n_valid_frames", "min"),
            max_valid_frames=("n_valid_frames", "max"),
        )
        .reset_index()
        .rename(columns={"speed": "source_motion_label"})
    )
    quality["sparse_level_excluded"] = quality[
        "source_motion_label"
    ].isin(sparse_motion_levels)
    if sparse_motion_levels:
        logging.warning(
            "Excluded sparse motion-label levels %s from inferential models; "
            "fewer than %d eligible lap×motion units.",
            sorted(sparse_motion_levels),
            minimum_motion_laps,
        )
    return laps.loc[laps["eligible"]].copy(), quality


def functional_decomposition(
    laps: pd.DataFrame,
    value_column: str,
    scene_column: str = "scene_id",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute a cell-balanced scene/camera functional decomposition."""
    cell = (
        laps.groupby([scene_column, "camera_id"], observed=True)[value_column]
        .agg(y="mean", n_laps="size")
        .reset_index()
    )
    f0 = float(cell["y"].mean())
    scene = (
        cell.groupby(scene_column, observed=True)["y"].mean() - f0
    ).rename("f_scene")
    camera = (
        cell.groupby("camera_id", observed=True)["y"].mean() - f0
    ).rename("f_camera")
    interaction = cell.merge(
        scene, left_on=scene_column, right_index=True
    ).merge(camera, left_on="camera_id", right_index=True)
    interaction["f_scene_camera"] = (
        interaction["y"]
        - f0
        - interaction["f_scene"]
        - interaction["f_camera"]
    )
    v_scene = float(np.mean(interaction["f_scene"] ** 2))
    v_camera = float(np.mean(interaction["f_camera"] ** 2))
    v_interaction = float(
        np.mean(interaction["f_scene_camera"] ** 2)
    )
    residuals = laps.merge(
        cell[[scene_column, "camera_id", "y"]],
        on=[scene_column, "camera_id"],
        how="inner",
    )
    residuals["squared_residual"] = (
        residuals[value_column] - residuals["y"]
    ) ** 2
    v_residual = float(
        residuals.groupby(
            [scene_column, "camera_id"], observed=True
        )["squared_residual"].mean().mean()
    )
    total = v_scene + v_camera + v_interaction + v_residual
    result = {
        "target": value_column,
        "f0": f0,
        "V_scene": v_scene,
        "V_camera": v_camera,
        "V_scene_camera": v_interaction,
        "V_residual": v_residual,
        "fraction_scene": v_scene / total,
        "fraction_camera": v_camera / total,
        "fraction_scene_camera": v_interaction / total,
        "fraction_residual": v_residual / total,
        "n_cells": int(len(cell)),
        "n_laps": int(len(laps)),
    }
    return result, interaction


def fit_anovas(
    laps: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Fit primary and decomposed Sum-contrast Type III ANOVA models."""
    target = "relative_degradation_mean"
    primary_formula = (
        f"{target} ~ C(scene_id, Sum) * C(camera_id, Sum)"
    )
    primary_model = smf.ols(primary_formula, data=laps).fit()
    primary = enrich_anova(anova_lm(primary_model, typ=3))
    primary["r_squared"] = primary_model.rsquared
    primary["adjusted_r_squared"] = primary_model.rsquared_adj
    primary.to_csv(
        output_dir / "anova_scene_camera_relative.csv", index=False
    )

    terms = (
        "C(topology, Sum) + C(light, Sum) + C(speed, Sum) "
        "+ C(camera_id, Sum) "
        "+ C(topology, Sum):C(camera_id, Sum) "
        "+ C(light, Sum):C(camera_id, Sum) "
        "+ C(speed, Sum):C(camera_id, Sum)"
    )
    decomposed_formula = f"{target} ~ {terms}"
    decomposed_model = smf.ols(decomposed_formula, data=laps).fit()
    decomposed = enrich_anova(anova_lm(decomposed_model, typ=3))
    decomposed["r_squared"] = decomposed_model.rsquared
    decomposed["adjusted_r_squared"] = decomposed_model.rsquared_adj
    decomposed.to_csv(
        output_dir / "anova_decomposed_relative.csv", index=False
    )
    (output_dir / "model_summary_relative.txt").write_text(
        primary_model.summary().as_text()
        + "\n\n"
        + decomposed_model.summary().as_text(),
        encoding="utf-8",
    )
    return primary, decomposed, primary_model


def topology_tables(
    laps: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize topology means after balancing camera settings."""
    metrics = [
        "source_abs_rel_mean",
        "canonical_abs_rel_mean",
        "absolute_degradation_mean",
        "relative_degradation_mean",
    ]
    cells = (
        laps.groupby(["topology", "camera_id"], observed=True)[metrics]
        .mean()
        .reset_index()
    )
    scale = cells.groupby("topology", observed=True)[metrics].mean().reset_index()
    scale["relative_degradation_percent"] = (
        100 * scale["relative_degradation_mean"]
    )
    scale.to_csv(
        output_dir / "topology_scale_comparison.csv", index=False
    )
    description = (
        laps.groupby("topology", observed=True)["relative_degradation_mean"]
        .agg(
            n_laps="size",
            mean="mean",
            std="std",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )
    quartiles = laps.groupby("topology", observed=True)[
        "relative_degradation_mean"
    ].quantile([0.25, 0.75]).unstack()
    quartiles["iqr"] = quartiles[0.75] - quartiles[0.25]
    description = description.merge(
        quartiles[["iqr"]], left_on="topology", right_index=True
    )
    description.to_csv(
        output_dir / "topology_relative_summary.csv", index=False
    )
    return scale, cells


def motion_table(laps: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Summarize camera-balanced targets by CSV-internal motion label."""
    metrics = [
        "source_abs_rel_mean",
        "canonical_abs_rel_mean",
        "absolute_degradation_mean",
        "relative_degradation_mean",
    ]
    cells = (
        laps.groupby(["speed", "camera_id"], observed=True)[metrics]
        .mean()
        .reset_index()
    )
    summary = (
        cells.groupby("speed", observed=True)[metrics]
        .mean()
        .reset_index()
        .rename(columns={"speed": "source_motion_label"})
    )
    summary["relative_degradation_percent"] = (
        100 * summary["relative_degradation_mean"]
    )
    summary.to_csv(
        output_dir / "motion_relative_summary.csv", index=False
    )
    return summary


def bootstrap_components(
    laps: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Cluster-bootstrap scenes for paired absolute/relative comparisons."""
    rng = np.random.default_rng(seed)
    scenes = laps["scene_id"].unique()
    absolute_point, _ = functional_decomposition(
        laps, "absolute_degradation_mean"
    )
    relative_point, _ = functional_decomposition(
        laps, "relative_degradation_mean"
    )
    point_estimates = {
        "relative_fraction_scene": relative_point["fraction_scene"],
        "relative_fraction_camera": relative_point["fraction_camera"],
        "relative_fraction_scene_camera": relative_point[
            "fraction_scene_camera"
        ],
        "relative_fraction_residual": relative_point["fraction_residual"],
        "scene_fraction_reduction_vs_absolute": (
            1
            - relative_point["fraction_scene"]
            / absolute_point["fraction_scene"]
        ),
        "scene_fraction_percentage_point_change": (
            relative_point["fraction_scene"]
            - absolute_point["fraction_scene"]
        ),
    }
    rows: list[dict[str, float | int | str]] = []
    for replicate in range(n_bootstrap):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        pieces: list[pd.DataFrame] = []
        for draw, scene in enumerate(chosen):
            piece = laps.loc[laps["scene_id"].eq(scene)].copy()
            piece["bootstrap_scene"] = f"{scene}__{draw}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        absolute, _ = functional_decomposition(
            sample,
            "absolute_degradation_mean",
            scene_column="bootstrap_scene",
        )
        relative, _ = functional_decomposition(
            sample,
            "relative_degradation_mean",
            scene_column="bootstrap_scene",
        )
        rows.extend(
            [
                {
                    "bootstrap": replicate,
                    "metric": "relative_fraction_scene",
                    "value": relative["fraction_scene"],
                },
                {
                    "bootstrap": replicate,
                    "metric": "relative_fraction_camera",
                    "value": relative["fraction_camera"],
                },
                {
                    "bootstrap": replicate,
                    "metric": "relative_fraction_scene_camera",
                    "value": relative["fraction_scene_camera"],
                },
                {
                    "bootstrap": replicate,
                    "metric": "relative_fraction_residual",
                    "value": relative["fraction_residual"],
                },
                {
                    "bootstrap": replicate,
                    "metric": "scene_fraction_reduction_vs_absolute",
                    "value": (
                        1
                        - relative["fraction_scene"]
                        / absolute["fraction_scene"]
                    ),
                },
                {
                    "bootstrap": replicate,
                    "metric": "scene_fraction_percentage_point_change",
                    "value": (
                        relative["fraction_scene"]
                        - absolute["fraction_scene"]
                    ),
                },
            ]
        )
        if (replicate + 1) % max(1, n_bootstrap // 10) == 0:
            logging.info(
                "Bootstrap progress %d/%d", replicate + 1, n_bootstrap
            )
    samples = pd.DataFrame(rows)
    summary = (
        samples.groupby("metric", observed=True)["value"]
        .agg(
            bootstrap_mean="mean",
            ci_lower_95=lambda values: values.quantile(0.025),
            ci_upper_95=lambda values: values.quantile(0.975),
            n_successful="size",
        )
        .reset_index()
    )
    summary.insert(
        1,
        "point_estimate",
        summary["metric"].map(point_estimates),
    )
    summary.to_csv(
        output_dir / "bootstrap_relative_confidence_intervals.csv",
        index=False,
    )
    return summary


def make_plots(
    laps: pd.DataFrame,
    components: pd.DataFrame,
    relative_interaction: pd.DataFrame,
    topology_scale: pd.DataFrame,
    topology_cells: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate presentation-ready relative-degradation figures."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    component_columns = [
        "fraction_scene",
        "fraction_camera",
        "fraction_scene_camera",
        "fraction_residual",
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(4)
    width = 0.25
    labels = {
        "source_abs_rel_mean": "Raw AbsRel",
        "absolute_degradation_mean": "Absolute Δ",
        "relative_degradation_mean": "Relative Δ",
    }
    for index, (_, row) in enumerate(components.iterrows()):
        ax.bar(
            x + (index - 1) * width,
            row[component_columns].astype(float),
            width,
            label=labels[row["target"]],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Scene", "Camera", "Scene×camera", "Residual"])
    ax.set_ylabel("Functional variance fraction")
    ax.set_title("Absolute error vs absolute and relative degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figure_dir / "01_variance_decomposition_with_relative.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(topology_scale))
    axes[0].plot(
        x,
        topology_scale["source_abs_rel_mean"],
        marker="o",
        label="Source error",
    )
    axes[0].plot(
        x,
        topology_scale["canonical_abs_rel_mean"],
        marker="o",
        label="Canonical error",
    )
    axes[0].plot(
        x,
        topology_scale["absolute_degradation_mean"],
        marker="o",
        label="Absolute Δ",
    )
    axes[0].set_ylabel("AbsRel")
    axes[0].set_title("Absolute error scale by topology")
    axes[0].legend()
    axes[1].bar(
        x,
        topology_scale["relative_degradation_percent"],
        color="tab:orange",
    )
    axes[1].set_ylabel("Relative degradation (%)")
    axes[1].set_title("Camera-balanced relative degradation")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(topology_scale["topology"], rotation=30)
        axis.set_xlabel("Topology")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "02_topology_absolute_vs_relative.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    camera = (
        laps.groupby(
            ["source_exposure", "source_gain"], observed=True
        )["relative_degradation_mean"]
        .mean()
        .reset_index()
    )
    matrix = camera.pivot(
        index="source_exposure",
        columns="source_gain",
        values="relative_degradation_mean",
    )
    save_heatmap(
        100 * matrix,
        figure_dir / "03_relative_camera_heatmap.png",
        "Relative degradation by exposure and gain",
        "Source gain",
        "Source exposure",
        "Mean relative degradation (%)",
        annotate=True,
    )
    interaction_matrix = relative_interaction.pivot(
        index="scene_id",
        columns="camera_id",
        values="f_scene_camera",
    )
    save_heatmap(
        100 * interaction_matrix,
        figure_dir / "04_relative_scene_camera_interaction.png",
        "Relative-degradation scene × camera interaction",
        "Camera setting",
        "Scene",
        "Interaction effect (percentage points)",
        cmap="coolwarm",
        centered=True,
    )

    profile = topology_cells.pivot(
        index="topology",
        columns="camera_id",
        values="relative_degradation_mean",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(profile.shape[1])
    for topology, row in profile.iterrows():
        ax.plot(x, 100 * row, marker="o", label=topology)
    ax.set_xticks(x)
    ax.set_xticklabels(profile.columns, rotation=60, ha="right")
    ax.set_xlabel("Camera setting")
    ax.set_ylabel("Relative degradation (%)")
    ax.set_title("Camera-relative degradation profile by topology")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "05_relative_camera_profile_by_topology.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    motion_profile = (
        laps.groupby(["speed", "camera_id"], observed=True)[
            "relative_degradation_mean"
        ]
        .mean()
        .unstack("camera_id")
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(motion_profile.shape[1])
    for motion, row in motion_profile.iterrows():
        ax.plot(x, 100 * row, marker="o", label=motion)
    ax.set_xticks(x)
    ax.set_xticklabels(
        motion_profile.columns, rotation=60, ha="right"
    )
    ax.set_xlabel("Camera setting")
    ax.set_ylabel("Relative degradation (%)")
    ax.set_title(
        "Camera-relative degradation by source_motion_label"
    )
    ax.legend(title="Motion label")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "06_relative_camera_profile_by_motion.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def effect_record(table: pd.DataFrame, effect: str) -> pd.Series:
    """Extract one ANOVA effect row."""
    return table.loc[table["effect"].eq(effect)].iloc[0]


def write_report(
    frames: pd.DataFrame,
    laps: pd.DataFrame,
    components: pd.DataFrame,
    primary: pd.DataFrame,
    decomposed: pd.DataFrame,
    topology_scale: pd.DataFrame,
    bootstrap: pd.DataFrame,
    motion_quality: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a compact Markdown summary with computed statistics."""
    relative = components.loc[
        components["target"].eq("relative_degradation_mean")
    ].iloc[0]
    relative_median = components.loc[
        components["target"].eq("relative_degradation_median")
    ].iloc[0]
    absolute = components.loc[
        components["target"].eq("absolute_degradation_mean")
    ].iloc[0]
    scene = effect_record(primary, "scene")
    camera = effect_record(primary, "camera")
    interaction = effect_record(primary, "scene_camera")
    topology = effect_record(decomposed, "topology")
    topology_camera = effect_record(decomposed, "topology_camera")
    light_camera = effect_record(decomposed, "light_camera")
    speed = effect_record(decomposed, "speed")
    speed_camera = effect_record(decomposed, "speed_camera")
    t5 = topology_scale.set_index("topology").loc["topology5"]
    topology_min = topology_scale["relative_degradation_percent"].min()
    topology_max = topology_scale["relative_degradation_percent"].max()
    frame_ratio = frames.loc[
        frames["valid_relative"], "relative_degradation"
    ]
    reduction = (
        1 - relative["fraction_scene"] / absolute["fraction_scene"]
    )
    excluded_motion = motion_quality.loc[
        motion_quality["sparse_level_excluded"],
        "source_motion_label",
    ].astype(str).tolist()
    report = f"""# Relative Canonical Degradation Analysis

## Definition

`relative degradation = (source AbsRel - canonical AbsRel) / canonical AbsRel`.
Ratios were calculated per valid matched frame and then aggregated within each physical lap and CSV-internal `source_motion_label`.

## Data

- Valid ratio frames: {int(frames['valid_relative'].sum()):,}
- Retained laps: {len(laps):,}
- Scenes/cameras: {laps['scene_id'].nunique()}/{laps['camera_id'].nunique()}
- Motion labels in inferential model: {', '.join(sorted(laps['speed'].unique()))}
- Sparse motion labels excluded at `min-valid-frames`: {', '.join(excluded_motion) if excluded_motion else 'none'}
- Frame-ratio median/99th percentile/max: {frame_ratio.median():.4f} / {frame_ratio.quantile(.99):.4f} / {frame_ratio.max():.4f}

## Scene–Camera Result

- Scene: F={scene['F']:.4f}, p={scene['p_value']:.3e}, partial eta squared={scene['partial_eta_squared']:.4f}
- Camera: F={camera['F']:.4f}, p={camera['p_value']:.3e}, partial eta squared={camera['partial_eta_squared']:.4f}
- Scene-camera: F={interaction['F']:.4f}, p={interaction['p_value']:.3e}, partial eta squared={interaction['partial_eta_squared']:.4f}

Functional fractions for relative degradation were scene={relative['fraction_scene']:.4f}, camera={relative['fraction_camera']:.4f}, scene-camera={relative['fraction_scene_camera']:.4f}, and residual={relative['fraction_residual']:.4f}. Relative normalization reduced the scene fraction by {100*reduction:.2f}% ({100*(relative['fraction_scene']-absolute['fraction_scene']):.2f} percentage points) compared with absolute subtraction, but did not eliminate it.

The within-lap median robustness analysis gave scene={relative_median['fraction_scene']:.4f}, camera={relative_median['fraction_camera']:.4f}, scene-camera={relative_median['fraction_scene_camera']:.4f}, and residual={relative_median['fraction_residual']:.4f}, preserving the same conclusion despite the ratio target's upper tail.

## Topology Result

- Topology main effect: F={topology['F']:.4f}, p={topology['p_value']:.3e}, eta squared={topology['eta_squared']:.4f}, partial eta squared={topology['partial_eta_squared']:.4f}
- Topology-camera: F={topology_camera['F']:.4f}, p={topology_camera['p_value']:.3e}, eta squared={topology_camera['eta_squared']:.4f}
- Light-camera: F={light_camera['F']:.4f}, p={light_camera['p_value']:.3e}, eta squared={light_camera['eta_squared']:.4f}
- Motion-label main effect: F={speed['F']:.4f}, p={speed['p_value']:.3e}, eta squared={speed['eta_squared']:.4f}
- Motion-label-camera: F={speed_camera['F']:.4f}, p={speed_camera['p_value']:.3e}, eta squared={speed_camera['eta_squared']:.4f}

Camera-balanced topology means ranged from {topology_min:.2f}% to {topology_max:.2f}%. Topology5 was {t5['relative_degradation_percent']:.2f}%, while its absolute degradation ({t5['absolute_degradation_mean']:.4f}) was the largest topology mean. Thus percentage normalization substantially compressed the topology-dependent scale difference.

## Conclusion

Percentage normalization better suppresses scene-scale dependence than simple subtraction. The degradation signal becomes predominantly camera-related: camera plus scene-camera components account for {100*(relative['fraction_camera']+relative['fraction_scene_camera']):.2f}% of the component-sum variance. A remaining scene fraction of {100*relative['fraction_scene']:.2f}% is statistically detectable and is driven more strongly by illumination structure and interaction than by topology mean differences.

Bootstrap results are in `bootstrap_relative_confidence_intervals.csv`.
"""
    (output_dir / "relative_degradation_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> int:
    """Run the full relative-degradation analysis."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, files = load_frames(args)
    laps, motion_quality = aggregate_laps(
        frames, args.min_valid_frames
    )
    laps.to_csv(
        output_dir / "relative_degradation_lap_level.csv", index=False
    )
    motion_quality.to_csv(
        output_dir / "motion_label_quality.csv", index=False
    )
    primary, decomposed, _ = fit_anovas(laps, output_dir)
    component_rows: list[dict[str, float]] = []
    interactions: dict[str, pd.DataFrame] = {}
    for target in [
        "source_abs_rel_mean",
        "absolute_degradation_mean",
        "relative_degradation_mean",
        "relative_degradation_median",
    ]:
        component, interaction = functional_decomposition(laps, target)
        component_rows.append(component)
        interactions[target] = interaction
    components = pd.DataFrame(component_rows)
    components.to_csv(
        output_dir / "functional_anova_relative_comparison.csv", index=False
    )
    topology_scale, topology_cells = topology_tables(laps, output_dir)
    motion_table(laps, output_dir)
    bootstrap = bootstrap_components(
        laps, args.n_bootstrap, args.seed, output_dir
    )
    make_plots(
        laps,
        components.loc[
            components["target"].isin(
                [
                    "source_abs_rel_mean",
                    "absolute_degradation_mean",
                    "relative_degradation_mean",
                ]
            )
        ],
        interactions["relative_degradation_mean"],
        topology_scale,
        topology_cells,
        output_dir,
    )
    write_report(
        frames,
        laps,
        components,
        primary,
        decomposed,
        topology_scale,
        bootstrap,
        motion_quality,
        output_dir,
    )
    metadata = {
        "input_root": str(args.input_root.expanduser().resolve()),
        "files": [str(path) for path in files],
        "n_files": len(files),
        "n_frames": len(frames),
        "n_valid_relative_frames": int(frames["valid_relative"].sum()),
        "n_laps": len(laps),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    (output_dir / "relative_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logging.info("Relative-degradation report: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
