from __future__ import annotations

import csv
import math
from glob import glob
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from comparison.gradient_base_unc import (
    GradientUncertaintyExtractor,
    invert_reference_depth,
    resolve_layer_names,
    transformed_reference_pixels,
)
from comparison.infer_noise import infer_noise_uncertainty, predict_depth

from dataset.ati_dataset_caminduce import (
    CameraParameterRange,
    FoundationCameraGroupedDataset,
    PairedResizeToTensor,
)
from evaluation_utils.eval_selection import (
    compute_selection_alpha_sweep,
    plot_selection_alpha_sweep,
)
from evaluation_utils.eval_metrics import compute_vector_masked_correlations
from model.dav2_ati_model import MODEL_IDS
from utils.train_utils import seed_everything, topology_id


PAIRWISE_METRIC_COLUMNS = [
    "auroc", "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "f1", "switch_rate", "coverage", "num_pairs",
    "tp", "fp", "tn", "fn",
]


class _DA3ImageProcessor:
    def __init__(self, process_res: int) -> None:
        from depth_anything_3.utils.io.input_processor import InputProcessor

        self.process_res = process_res
        self.processor = InputProcessor()

    def __call__(self, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("DA3 image processing only supports return_tensors='pt'.")
        pixel_values, _, _ = self.processor(
            images,
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
            num_workers=1,
            sequential=True,
        )
        return {"pixel_values": pixel_values}


def comparison_method(cfg: DictConfig) -> str:
    method = str(cfg.evaluation.get("method", cfg.model.get("method", "tta"))).lower()
    if method not in {"tta", "gradient"}:
        raise ValueError("evaluation.method must be either 'tta' or 'gradient'.")
    return method


def uncertainty_layer_names(
    model,
    cfg: DictConfig,
    *,
    is_dav3: bool,
    num_auto_layers: int,
) -> list[str]:
    configured = cfg.evaluation.get("gradient_layers")
    if configured:
        return [str(name) for name in configured]
    if is_dav3:
        return ["model.head.scratch.output_conv1"]
    return resolve_layer_names(model, ("auto",), num_auto_layers)


def gradient_uncertainty(
    model,
    extractor: GradientUncertaintyExtractor,
    images: torch.Tensor,
    *,
    is_dav3: bool,
    amp: bool,
    reference_transform: str,
    reference_noise_std: float,
    min_depth: float,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    extractor.clear()
    grad_images = images.detach().clone().requires_grad_(True)
    reference_images = transformed_reference_pixels(
        images.detach(), reference_transform, reference_noise_std
    )
    with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=amp):
        reference_depth = predict_depth(
            model,
            reference_images,
            is_dav3=is_dav3,
            target_size=tuple(images.shape[-2:]),
        )
        reference_depth = invert_reference_depth(
            reference_depth, reference_transform
        ).clamp_min(min_depth)
    with torch.autocast(device_type=images.device.type, enabled=amp):
        depth = predict_depth(
            model,
            grad_images,
            is_dav3=is_dav3,
            target_size=tuple(images.shape[-2:]),
        )
        auxiliary_loss = (depth.clamp_min(min_depth) - reference_depth).square().mean()
    auxiliary_loss.backward()
    return extractor.uncertainty_map(tuple(images.shape[-2:])).flatten(1).mean(1)


def uncertainty_correlation_rows(
    predictions: dict[str, torch.Tensor],
    split_masks: dict[str, torch.Tensor],
    method: str,
) -> list[dict]:
    score = predictions["camera_bias"]
    targets = {
        "abs_rel_degradation": predictions["abs_rel_degradation"],
        "candidate_abs_rel": predictions["candidate_abs_rel"],
    }
    rows = []
    for split, mask in split_masks.items():
        for target_name, target in targets.items():
            uncertainty = score[mask]
            values = target[mask]
            valid = torch.isfinite(uncertainty) & torch.isfinite(values)
            correlations = compute_vector_masked_correlations(
                uncertainty,
                values,
                valid_mask=valid,
                max_samples=None,
                prefix="correlation",
            )
            rows.append({
                "method": method,
                "split": split,
                "target": target_name,
                "pearson": correlations["correlation_pearson"],
                "spearman": correlations["correlation_spearman"],
                "num_samples": int(valid.sum()),
                "uncertainty_mean": (
                    float(uncertainty[valid].mean()) if valid.any() else float("nan")
                ),
                "target_mean": (
                    float(values[valid].mean()) if valid.any() else float("nan")
                ),
            })
    return rows


def mean_relative_selection_regret(
    predicted_score: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    group_id: torch.Tensor,
    min_settings_per_group: int,
    eps: float = 1e-8,
) -> float:
    predicted_score = predicted_score.detach().float().flatten()
    candidate_abs_rel = candidate_abs_rel.detach().float().flatten()
    group_id = group_id.detach().flatten()
    valid = (
        torch.isfinite(predicted_score)
        & torch.isfinite(candidate_abs_rel)
        & (candidate_abs_rel >= 0)
        & torch.isfinite(group_id.float())
    )
    regrets = []
    for value in torch.unique(group_id[valid]):
        mask = valid & (group_id == value)
        if int(mask.sum()) < min_settings_per_group:
            continue
        scores = predicted_score[mask]
        abs_rel = candidate_abs_rel[mask]
        optimal = abs_rel.min()
        selected = abs_rel[torch.argmin(scores)]
        regrets.append((selected - optimal).clamp_min(0) / optimal.clamp_min(eps) * 100.0)
    return float(torch.stack(regrets).mean()) if regrets else float("nan")


def mean_candidate_degradation(
    candidate_abs_rel: torch.Tensor,
    canonical_abs_rel: torch.Tensor,
    group_id: torch.Tensor,
    is_canonical_setting: torch.Tensor,
    min_settings_per_group: int,
    eps: float = 1e-8,
) -> dict[str, float]:
    candidate_abs_rel = candidate_abs_rel.detach().float().flatten()
    canonical_abs_rel = canonical_abs_rel.detach().float().flatten()
    group_id = group_id.detach().flatten()
    is_canonical_setting = is_canonical_setting.detach().bool().flatten()
    valid = (
        torch.isfinite(candidate_abs_rel)
        & torch.isfinite(canonical_abs_rel)
        & (candidate_abs_rel >= 0)
        & (canonical_abs_rel >= 0)
        & torch.isfinite(group_id.float())
    )
    absolute, relative = [], []
    for value in torch.unique(group_id[valid]):
        group_mask = valid & (group_id == value)
        if int(group_mask.sum()) < min_settings_per_group:
            continue
        candidate_mask = group_mask & ~is_canonical_setting
        if not candidate_mask.any():
            continue
        degradation = (
            candidate_abs_rel[candidate_mask] - canonical_abs_rel[candidate_mask]
        ).clamp_min(0)
        absolute.append(degradation.mean())
        relative.append(
            (degradation / canonical_abs_rel[candidate_mask].clamp_min(eps) * 100.0).mean()
        )
    return {
        "candidate_mean_degradation_abs_rel": (
            float(torch.stack(absolute).mean()) if absolute else float("nan")
        ),
        "candidate_mean_relative_degradation_percent": (
            float(torch.stack(relative).mean()) if relative else float("nan")
        ),
    }


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def beta_grid(cfg: DictConfig) -> list[float]:
    sweep = cfg.evaluation.pairwise_beta_sweep
    values = torch.linspace(
        float(sweep.min), float(sweep.max), int(sweep.num_points),
        dtype=torch.float64,
    ).tolist()
    return sorted(set(values + [float(cfg.pairwise_policy.beta)]))


def ordered_pairwise_data(
    mean: torch.Tensor,
    std: torch.Tensor,
    abs_rel: torch.Tensor,
    group_id: torch.Tensor,
    m_switch: float,
    tie_eps_percent: float,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor | float]:
    scores, labels, gaps = [], [], []
    for value in torch.unique(group_id):
        mask = group_id == value
        group_mean, group_std, group_abs_rel = mean[mask], std[mask], abs_rel[mask]
        pair_i, pair_j = torch.triu_indices(
            group_mean.numel(), group_mean.numel(), offset=1
        )
        valid = (
            torch.isfinite(group_mean[pair_i])
            & torch.isfinite(group_mean[pair_j])
            & torch.isfinite(group_std[pair_i])
            & torch.isfinite(group_std[pair_j])
            & torch.isfinite(group_abs_rel[pair_i])
            & torch.isfinite(group_abs_rel[pair_j])
            & (group_std[pair_i] >= 0)
            & (group_std[pair_j] >= 0)
            & (group_abs_rel[pair_i] >= 0)
            & (group_abs_rel[pair_j] >= 0)
        )
        pair_i, pair_j = pair_i[valid], pair_j[valid]
        if pair_i.numel() == 0:
            continue
        pair_std = torch.hypot(group_std[pair_i], group_std[pair_j])
        gap = (group_abs_rel[pair_i] - group_abs_rel[pair_j]).abs() / (
            torch.minimum(group_abs_rel[pair_i], group_abs_rel[pair_j]) + eps
        ) * 100.0
        scores.append(torch.cat([
            (group_mean[pair_i] - group_mean[pair_j] - m_switch) / (pair_std + eps),
            (group_mean[pair_j] - group_mean[pair_i] - m_switch) / (pair_std + eps),
        ]))
        labels.append(torch.cat([
            group_abs_rel[pair_j] < group_abs_rel[pair_i],
            group_abs_rel[pair_i] < group_abs_rel[pair_j],
        ]))
        gaps.append(torch.cat([gap, gap]))

    if not scores:
        empty = torch.empty(0)
        return {"scores": empty, "labels": empty.bool(), "coverage": float("nan")}
    scores, labels, gaps = torch.cat(scores), torch.cat(labels), torch.cat(gaps)
    keep = gaps >= tie_eps_percent
    return {
        "scores": scores[keep],
        "labels": labels[keep],
        "coverage": float(keep.float().mean()),
    }


def roc_curve(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not scores.numel() or not positives or not negatives:
        nan = scores.new_tensor(float("nan"))
        return nan[None], nan[None], float("nan")
    order = torch.argsort(scores, descending=True)
    scores, labels = scores[order], labels[order]
    ends = torch.cat([
        torch.nonzero(scores[:-1] != scores[1:], as_tuple=False).flatten(),
        torch.tensor([scores.numel() - 1]),
    ])
    tp = labels.long().cumsum(0)[ends].float()
    fp = (~labels).long().cumsum(0)[ends].float()
    tpr = torch.cat([torch.zeros(1), tp / positives])
    fpr = torch.cat([torch.zeros(1), fp / negatives])
    return fpr, tpr, float(torch.trapz(tpr, fpr))


def policy_metrics(
    pair_data: dict[str, torch.Tensor | float],
    beta: float,
    auroc: float,
) -> dict[str, float | int]:
    scores, labels = pair_data["scores"], pair_data["labels"]
    decisions = scores > beta
    tp = int((decisions & labels).sum())
    fp = int((decisions & ~labels).sum())
    tn = int((~decisions & ~labels).sum())
    fn = int((~decisions & labels).sum())
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1_denominator = 2 * tp + fp + fn
    return {
        "auroc": auroc,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": (
            float("nan") if math.isnan(precision)
            else 2 * tp / f1_denominator if f1_denominator else float("nan")
        ),
        "switch_rate": (tp + fp) / total if total else float("nan"),
        "coverage": pair_data["coverage"],
        "num_pairs": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def evaluate_pairwise_policy(
    predictions: dict[str, torch.Tensor],
    split_masks: dict[str, torch.Tensor],
    cfg: DictConfig,
) -> tuple[list[dict], list[dict], dict]:
    betas = beta_grid(cfg)
    tie_values = list(map(float, cfg.pairwise_policy.eval_tie_eps_percent))
    m_switch = float(cfg.pairwise_policy.m_switch)
    pair_data, sweep_rows = {}, []
    for split, mask in split_masks.items():
        for tie_eps in tie_values:
            data = ordered_pairwise_data(
                predictions["camera_bias"][mask], predictions["camera_std"][mask],
                predictions["candidate_abs_rel"][mask], predictions["group_id"][mask],
                m_switch, tie_eps,
            )
            pair_data[split, tie_eps] = data
            _, _, auroc = roc_curve(data["scores"], data["labels"])
            for beta in betas:
                sweep_rows.append({
                    "split": split,
                    "tie_eps_percent": tie_eps,
                    "beta": beta,
                    "switch_probability": normal_cdf(beta),
                    "m_switch": m_switch,
                    **policy_metrics(data, beta, auroc),
                })

    selection_split = str(cfg.evaluation.pairwise_beta_sweep.selection_split)
    if selection_split != "seen":
        raise ValueError("pairwise beta selection_split must be 'seen'.")
    objective = str(cfg.evaluation.pairwise_beta_sweep.objective)
    if objective not in PAIRWISE_METRIC_COLUMNS:
        raise ValueError(f"Unknown pairwise beta objective: {objective}")
    selected, summary_rows = {}, []
    for tie_eps in tie_values:
        candidates = [
            row for row in sweep_rows
            if row["split"] == selection_split and row["tie_eps_percent"] == tie_eps
        ]
        best = max(
            candidates,
            key=lambda row: (
                -math.inf if math.isnan(float(row[objective])) else float(row[objective]),
                -math.inf if math.isnan(float(row["precision"])) else float(row["precision"]),
                float(row["beta"]),
            ),
        )
        selected[tie_eps] = float(best["beta"])
        for split in split_masks:
            row = next(
                value for value in sweep_rows
                if value["split"] == split
                and value["tie_eps_percent"] == tie_eps
                and value["beta"] == best["beta"]
            )
            summary_rows.append({
                "tie_eps_percent": tie_eps,
                "selection_split": selection_split,
                "objective": objective,
                "selected_beta": best["beta"],
                "selected_switch_probability": best["switch_probability"],
                "evaluation_split": split,
                **{key: row[key] for key in PAIRWISE_METRIC_COLUMNS},
            })
    return sweep_rows, summary_rows, {"pairs": pair_data, "selected": selected}


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_pairwise_policy(
    split: str,
    sweep_rows: list[dict],
    plot_data: dict,
    configured_beta: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    roc_axis, accuracy_axis, precision_axis, recall_axis = axes.flatten()
    tie_values = sorted(plot_data["selected"])
    colors = plt.get_cmap("tab10").colors
    for index, tie_eps in enumerate(tie_values):
        color = colors[index]
        data = plot_data["pairs"][split, tie_eps]
        fpr, tpr, auroc = roc_curve(data["scores"], data["labels"])
        roc_axis.plot(fpr, tpr, color=color, label=f"tie<{tie_eps:g}% AUROC={auroc:.3f}")
        rows = [
            row for row in sweep_rows
            if row["split"] == split and row["tie_eps_percent"] == tie_eps
        ]
        betas = [row["beta"] for row in rows]
        for axis, metric in (
            (accuracy_axis, "accuracy"),
            (precision_axis, "precision"),
            (recall_axis, "recall"),
        ):
            axis.plot(betas, [row[metric] for row in rows], color=color, label=f"tie<{tie_eps:g}%")
            selected_beta = plot_data["selected"][tie_eps]
            axis.axvline(selected_beta, color=color, linestyle="-", alpha=0.55)

    roc_axis.plot([0, 1], [0, 1], "k--", alpha=0.4)
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    roc_axis.legend()
    accuracy_axis.axhline(0.5, color="gray", linestyle=":", label="always stay")
    for axis, title, ylabel in (
        (accuracy_axis, "Accuracy vs beta", "Accuracy"),
        (precision_axis, "Precision vs beta", "Precision"),
        (recall_axis, "Recall vs beta", "Recall"),
    ):
        axis.axvline(configured_beta, color="black", linestyle="--", label="configured beta")
        axis.set(xlabel="beta", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    probability_note = (
        f"configured: beta={configured_beta:g}, Phi={normal_cdf(configured_beta):.4f}\n"
        + ", ".join(
            f"tie {tie:g}%: beta={plot_data['selected'][tie]:.4g}, "
            f"Phi={normal_cdf(plot_data['selected'][tie]):.4f}"
            for tie in tie_values
        )
    )
    figure.text(0.5, 0.01, probability_note, ha="center", fontsize="small")
    figure.suptitle(f"Pairwise policy — {split}")
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def build_validation_dataset(
    cfg: DictConfig,
    image_processor,
) -> FoundationCameraGroupedDataset:
    csv_root = Path(to_absolute_path(str(cfg.evaluation.csv_path)))
    csv_paths = (
        [str(csv_root)]
        if csv_root.is_file()
        else sorted(glob(str(csv_root / "*.csv")))
    )
    if not csv_paths:
        raise ValueError(f"No CSV files found in {csv_root}")

    dataset_root = str(
        Path(to_absolute_path(str(cfg.evaluation.dataset_root)))
    )
    return FoundationCameraGroupedDataset(
        csv_paths=csv_paths,
        foundation_model_name=cfg.model.model_id,
        camera_model_name=cfg.model.camera_model_name,
        parameter_range=CameraParameterRange(
            exposure_min=cfg.dataset.exposure_min,
            exposure_max=cfg.dataset.exposure_max,
            gain_min=cfg.dataset.gain_min,
            gain_max=cfg.dataset.gain_max,
        ),
        candidates_per_group=cfg.evaluation.min_camera_settings,
        candidate_sampling="parameter_diverse",
        parameter_normalization="linear",
        context_output_range="zero_one",
        path_replacements={
            "/dataset/ATI/MDE/orbbec_realworld_dataset":
                dataset_root,
            "/datasets/ATI/MDE/orbbec_realworld_dataset":
                dataset_root,
            "/media/michael/ssd1/AIoT_ATI/orbbec_realworld_dataset":
                dataset_root,
        },
        pair_transform=PairedResizeToTensor(
            image_processor=image_processor,
        ),
        include_canonical_setting_as_candidate=True,
        min_overlap_ratio=cfg.dataset.min_registration_overlap_ratio,
        min_ecc_score=cfg.dataset.min_registration_ecc_score,
        max_time_diff_sec=cfg.dataset.max_pair_time_diff_sec,
        max_registration_translation_px=(
            cfg.dataset.max_registration_translation_px
        ),
        abs_rel_degradation_quantile=None,
        use_all_candidates=True,
        topologies=(
            list(cfg.dataset.seen_val_topologies)
            + list(cfg.dataset.unseen_val_topologies)
        ),
        load_images=True,
        load_depth=False,
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        seed=int(cfg.get("training", {}).get("seed", 0)),
    )


def collect_predictions(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    cfg: DictConfig,
    method: str,
    is_dav3: bool,
    amp: bool,
    inference_batch_size: int,
) -> dict[str, torch.Tensor]:
    collected = {
        "camera_bias": [],
        "camera_std": [],
        "candidate_abs_rel": [],
        "canonical_abs_rel": [],
        "abs_rel_degradation": [],
        "is_canonical_setting": [],
        "group_id": [],
        "topology": [],
    }
    model.eval()
    layer_names = (
        uncertainty_layer_names(
            model,
            cfg,
            is_dav3=is_dav3,
            num_auto_layers=int(cfg.evaluation.get("gradient_num_auto_layers", 4)),
        )
        if method == "gradient" else []
    )
    extractor = (
        GradientUncertaintyExtractor(
            model,
            layer_names,
            channel_reduction=str(cfg.evaluation.get("gradient_channel_reduction", "max_abs")),
            layer_reduction=str(cfg.evaluation.get("gradient_layer_reduction", "max")),
        )
        if method == "gradient" else None
    )

    try:
        for batch in tqdm(
            loader,
            desc=f"Selection inference ({method})",
            dynamic_ncols=True,
        ):
            candidate_images = batch["candidate_images"].squeeze(0)
            group_size = candidate_images.shape[0]
            uncertainty_chunks = []
            for start in range(0, group_size, inference_batch_size):
                images = candidate_images[start:start + inference_batch_size].to(
                    device=device,
                    non_blocking=True,
                )
                if method == "tta":
                    with torch.autocast(device_type=device.type, enabled=amp):
                        result = infer_noise_uncertainty(
                            model,
                            images,
                            is_dav3=is_dav3,
                            sample_times=int(cfg.evaluation.get("tta_sample_times", 32)),
                            noise_std=float(cfg.evaluation.get("tta_noise_std", 1.0)),
                        )
                    uncertainty = result["image_uncertainty"]
                else:
                    uncertainty = gradient_uncertainty(
                        model,
                        extractor,
                        images,
                        is_dav3=is_dav3,
                        amp=amp,
                        reference_transform=str(
                            cfg.evaluation.get("gradient_reference_transform", "flip")
                        ),
                        reference_noise_std=float(
                            cfg.evaluation.get("gradient_reference_noise_std", 0.01)
                        ),
                        min_depth=float(cfg.dataset.min_depth),
                    )
                uncertainty_chunks.append(uncertainty.detach().float().cpu())

            uncertainty = torch.cat(uncertainty_chunks)
            collected["camera_bias"].append(uncertainty)
            collected["camera_std"].append(torch.zeros_like(uncertainty))
            collected["candidate_abs_rel"].append(
                batch["candidate_abs_rel"].flatten().float()
            )
            collected["canonical_abs_rel"].append(
                batch["canonical_abs_rel"].flatten().float()
            )
            collected["abs_rel_degradation"].append(
                batch["abs_rel_degradation"].flatten().float()
            )
            collected["is_canonical_setting"].append(
                (batch["candidate_exposure"].flatten() == batch["canonical_exposure"].flatten()[0])
                & (batch["candidate_gain"].flatten() == batch["canonical_gain"].flatten()[0])
            )
            collected["group_id"].append(
                batch["group_index"].flatten().repeat_interleave(group_size)
            )
            collected["topology"].append(
                batch["info"][:, 6].flatten().repeat_interleave(group_size)
            )
    finally:
        if extractor is not None:
            extractor.remove()

    return {
        key: torch.cat(values).flatten()
        for key, values in collected.items()
    }


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="comparison_caminduce",
)
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get("training", {}).get("seed", 0))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not bool(cfg.evaluation.get("no_amp", False))
    method = comparison_method(cfg)
    model_key = str(cfg.model.model_id)
    model_id = MODEL_IDS.get(model_key, model_key)
    is_dav3 = model_key.lower().startswith("da3") or "/da3" in model_key.lower()

    image_processor = (
        _DA3ImageProcessor(int(cfg.model.get("dav3_process_res", 504)))
        if is_dav3
        else AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    )
    dataset = build_validation_dataset(cfg, image_processor)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=device.type == "cuda",
    )

    if is_dav3:
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(model_id)
    else:
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    predictions = collect_predictions(
        model,
        loader,
        device,
        cfg=cfg,
        method=method,
        is_dav3=is_dav3,
        amp=amp,
        inference_batch_size=cfg.evaluation.inference_batch_size,
    )
    seen_topologies = {
        topology_id(value)
        for value in cfg.dataset.seen_val_topologies
    }
    unseen_topologies = {
        topology_id(value)
        for value in cfg.dataset.unseen_val_topologies
    }
    topology_values = predictions["topology"].long()
    split_masks = {
        "all": torch.ones_like(topology_values, dtype=torch.bool),
        "seen": torch.isin(
            topology_values,
            torch.tensor(sorted(seen_topologies)),
        ),
        "unseen": torch.isin(
            topology_values,
            torch.tensor(sorted(unseen_topologies)),
        ),
    }
    for topology_number in sorted(
        topology_values.unique().tolist()
    ):
        split_masks[f"topology{topology_number}"] = (
            topology_values == topology_number
        )

    sweeps = {}
    csv_rows = []
    for split_name, split_mask in split_masks.items():
        candidate_degradation = mean_candidate_degradation(
            predictions["candidate_abs_rel"][split_mask],
            predictions["canonical_abs_rel"][split_mask],
            predictions["group_id"][split_mask],
            predictions["is_canonical_setting"][split_mask],
            cfg.evaluation.min_camera_settings,
        )
        rows = compute_selection_alpha_sweep(
            predictions["camera_bias"][split_mask],
            predictions["camera_std"][split_mask],
            predictions["candidate_abs_rel"][split_mask],
            predictions["group_id"][split_mask],
            cfg.evaluation.alpha_sweep_values,
            min_settings_per_group=cfg.evaluation.min_camera_settings,
            relative_regret_thresholds=(
                cfg.evaluation.relative_regret_thresholds_percent
            ),
            top_k_thresholds=(1, 3, 5),
        )
        for row in rows:
            row["method"] = method
            row["selection_mean_relative_regret_percent"] = mean_relative_selection_regret(
                predictions["camera_bias"][split_mask]
                + float(row["alpha"]) * predictions["camera_std"][split_mask],
                predictions["candidate_abs_rel"][split_mask],
                predictions["group_id"][split_mask],
                cfg.evaluation.min_camera_settings,
            )
            row.update(candidate_degradation)
        sweeps[split_name] = rows
        csv_rows.extend(
            {"split": split_name, **row}
            for row in rows
        )

    output_dir = Path(to_absolute_path(str(cfg.evaluation.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "selection_performance.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(csv_rows[0]),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    correlation_rows = uncertainty_correlation_rows(
        predictions,
        split_masks,
        method,
    )
    correlation_csv_path = output_dir / "uncertainty_correlations.csv"
    write_rows(
        correlation_csv_path,
        [
            "method", "split", "target", "pearson", "spearman",
            "num_samples", "uncertainty_mean", "target_mean",
        ],
        correlation_rows,
    )

    figure = plot_selection_alpha_sweep(
        {
            split_name: sweeps[split_name]
            for split_name in ("all", "seen", "unseen")
        }
    )
    relative_regret_axis = figure.axes[1].twinx()
    for split_name in ("all", "seen", "unseen"):
        rows = sweeps[split_name]
        relative_regret_axis.plot(
            [row["alpha"] for row in rows],
            [row["selection_mean_relative_regret_percent"] for row in rows],
            linestyle="--",
            marker="x",
            label=f"{split_name} (relative)",
        )
    relative_regret_axis.set_ylabel("mean relative regret vs optimal (%)")
    relative_regret_axis.legend(loc="upper right", fontsize="small")
    curve_path = output_dir / "selection_alpha_sweep.png"
    figure.savefig(curve_path, dpi=200)
    import matplotlib.pyplot as plt
    plt.close(figure)

    pairwise_rows, pairwise_summary, pairwise_plot_data = evaluate_pairwise_policy(
        predictions, split_masks, cfg
    )
    for row in pairwise_rows + pairwise_summary:
        row["method"] = method
    pairwise_csv_path = output_dir / "pairwise_beta_sweep.csv"
    write_rows(
        pairwise_csv_path,
        [
            "method", "split", "tie_eps_percent", "beta", "switch_probability",
            "m_switch", *PAIRWISE_METRIC_COLUMNS,
        ],
        pairwise_rows,
    )
    pairwise_summary_path = output_dir / "pairwise_beta_summary.csv"
    write_rows(
        pairwise_summary_path,
        [
            "method", "tie_eps_percent", "selection_split", "objective", "selected_beta",
            "selected_switch_probability", "evaluation_split",
            *PAIRWISE_METRIC_COLUMNS,
        ],
        pairwise_summary,
    )
    pairwise_plot_dir = output_dir / "pairwise_policy"
    pairwise_plot_dir.mkdir(parents=True, exist_ok=True)
    for split in ("seen", "unseen"):
        plot_pairwise_policy(
            split,
            pairwise_rows,
            pairwise_plot_data,
            float(cfg.pairwise_policy.beta),
            pairwise_plot_dir / f"pairwise_policy_{split}.png",
        )

    print(f"method: {method}")
    print(f"depth model: {model_id}")
    print(f"validation groups: {len(dataset):,}")
    print(f"selection CSV: {csv_path}")
    print(f"uncertainty correlation CSV: {correlation_csv_path}")
    print(f"alpha-sweep curve: {curve_path}")
    print(f"pairwise beta sweep: {pairwise_csv_path}")
    print(f"pairwise beta summary: {pairwise_summary_path}")
    print(f"pairwise policy figures: {pairwise_plot_dir}")


if __name__ == "__main__":
    main()
