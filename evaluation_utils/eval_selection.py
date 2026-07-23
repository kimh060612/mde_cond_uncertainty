from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


DEFAULT_RELATIVE_REGRET_THRESHOLDS = (3.0, 5.0, 10.0)


def _threshold_metric_name(threshold: float) -> str:
    value = f"{float(threshold):g}".replace(".", "p")
    return f"selection_accuracy_within_{value}pct"


def compute_selection_metrics(
    predicted_score: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    group_id: torch.Tensor,
    *,
    min_settings_per_group: int = 10,
    relative_regret_thresholds: Sequence[float] = DEFAULT_RELATIVE_REGRET_THRESHOLDS,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Evaluate camera-setting selection within each canonical group.

    Lower predicted scores and lower AbsRel values are better. ``selected
    regret`` is the absolute AbsRel gap between the selected and oracle
    settings. Percentage-tolerant accuracy uses that gap relative to the
    oracle AbsRel.
    """
    if min_settings_per_group < 2:
        raise ValueError("min_settings_per_group must be at least 2")

    predicted_score = predicted_score.detach().float().flatten()
    candidate_abs_rel = candidate_abs_rel.detach().float().flatten()
    group_id = group_id.detach().flatten()
    if not (
        predicted_score.shape
        == candidate_abs_rel.shape
        == group_id.shape
    ):
        raise ValueError(
            "predicted_score, candidate_abs_rel, and group_id must have "
            "the same flattened shape"
        )

    thresholds = tuple(float(value) for value in relative_regret_thresholds)
    if any(value <= 0 for value in thresholds):
        raise ValueError("relative_regret_thresholds must contain positive values")

    valid_mask = (
        torch.isfinite(predicted_score)
        & torch.isfinite(candidate_abs_rel)
        & (candidate_abs_rel >= 0)
        & torch.isfinite(group_id.float())
    )
    exact_hits: list[float] = []
    selected_regrets: list[float] = []
    tolerant_hits = {threshold: [] for threshold in thresholds}

    for current_group_id in torch.unique(group_id[valid_mask]):
        group_mask = valid_mask & (group_id == current_group_id)
        if int(group_mask.sum().item()) < min_settings_per_group:
            continue

        group_score = predicted_score[group_mask]
        group_abs_rel = candidate_abs_rel[group_mask]
        selected_abs_rel = group_abs_rel[torch.argmin(group_score)]
        oracle_abs_rel = group_abs_rel.min()
        selected_regret = (selected_abs_rel - oracle_abs_rel).clamp_min(0)
        relative_regret = selected_regret / oracle_abs_rel.clamp_min(eps) * 100.0

        exact_hits.append(
            float(
                torch.isclose(
                    selected_abs_rel,
                    oracle_abs_rel,
                    rtol=1e-6,
                    atol=eps,
                ).item()
            )
        )
        selected_regrets.append(float(selected_regret.item()))
        relative_regret_value = float(relative_regret.item())
        for threshold in thresholds:
            tolerant_hits[threshold].append(
                float(relative_regret_value < threshold)
            )

    metric_names = [_threshold_metric_name(value) for value in thresholds]
    if not exact_hits:
        return {
            "num_groups": 0.0,
            "selection_accuracy": float("nan"),
            **{name: float("nan") for name in metric_names},
            "selection_mean_regret_abs_rel": float("nan"),
        }

    return {
        "num_groups": float(len(exact_hits)),
        "selection_accuracy": sum(exact_hits) / len(exact_hits),
        **{
            _threshold_metric_name(threshold): (
                sum(tolerant_hits[threshold]) / len(tolerant_hits[threshold])
            )
            for threshold in thresholds
        },
        "selection_mean_regret_abs_rel": (
            sum(selected_regrets) / len(selected_regrets)
        ),
    }


def compute_selection_alpha_sweep(
    camera_bias: torch.Tensor,
    camera_std: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    group_id: torch.Tensor,
    alpha_values: Sequence[float],
    *,
    min_settings_per_group: int = 10,
    relative_regret_thresholds: Sequence[float] = DEFAULT_RELATIVE_REGRET_THRESHOLDS,
) -> list[dict[str, float]]:
    camera_bias = camera_bias.detach().float().flatten()
    camera_std = camera_std.detach().float().flatten()
    if camera_bias.shape != camera_std.shape:
        raise ValueError("camera_bias and camera_std must have the same shape")

    rows: list[dict[str, float]] = []
    for alpha in alpha_values:
        alpha = float(alpha)
        rows.append(
            {
                "alpha": alpha,
                **compute_selection_metrics(
                    camera_bias + alpha * camera_std,
                    candidate_abs_rel,
                    group_id,
                    min_settings_per_group=min_settings_per_group,
                    relative_regret_thresholds=relative_regret_thresholds,
                ),
            }
        )
    return rows


def plot_selection_alpha_sweep(
    sweeps: Mapping[str, Sequence[Mapping[str, Any]]],
):
    """Create one compact figure for W&B and standalone evaluation."""
    import matplotlib.pyplot as plt

    figure, (accuracy_axis, regret_axis) = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        constrained_layout=True,
    )
    for split_name, rows in sweeps.items():
        if not rows:
            continue
        alpha = [float(row["alpha"]) for row in rows]
        accuracy = [float(row["selection_accuracy"]) for row in rows]
        regret = [float(row["selection_mean_regret_abs_rel"]) for row in rows]
        accuracy_axis.plot(alpha, accuracy, marker="o", label=split_name)
        regret_axis.plot(alpha, regret, marker="o", label=split_name)

    accuracy_axis.set(
        xlabel="alpha",
        ylabel="selection accuracy",
        title="Camera-setting selection accuracy",
        ylim=(-0.02, 1.02),
    )
    regret_axis.set(
        xlabel="alpha",
        ylabel="mean selected regret (AbsRel)",
        title="Selected regret",
    )
    for axis in (accuracy_axis, regret_axis):
        axis.grid(alpha=0.3)
        axis.legend()
    return figure
