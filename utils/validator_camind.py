from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.ati_dataset_caminduce import flatten_group_batch
from evaluation_utils.eval_metrics import compute_vector_masked_correlations
from model.loss_fn import (
    scalar_heteroscedastic_loss,
    scale_shift_invariant_depth_loss,
    signed_pairwise_ranknet_loss,
    scalar_heteroscedastic_laplace_loss,
    log_scale_invariant_depth_difference
)
from utils.train_utils import reshape_group_batch, tensor_device


_DEGRADATION_EPS = 1e-6
_NEAR_ORACLE_THRESHOLDS = (5.0, 10.0, 20.0)


def _finite_mean(values: torch.Tensor) -> float:
    values = values.detach().float().flatten()
    finite_mask = torch.isfinite(values)
    if not finite_mask.any():
        return float("nan")
    return float(values[finite_mask].mean().item())


def _pairwise_rank_accuracy(
    predicted_score: torch.Tensor,
    target_score: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    _, num_candidates = predicted_score.shape
    if num_candidates < 2:
        return float("nan")

    pair_i, pair_j = torch.triu_indices(
        num_candidates,
        num_candidates,
        offset=1,
        device=predicted_score.device,
    )
    pred_diff = predicted_score[:, pair_i] - predicted_score[:, pair_j]
    target_diff = target_score[:, pair_i] - target_score[:, pair_j]
    pair_valid_mask = (
        torch.isfinite(pred_diff)
        & torch.isfinite(target_diff)
        & (target_diff != 0)
    )
    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if valid_mask.shape != predicted_score.shape:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} does not match "
                f"predicted_score shape {tuple(predicted_score.shape)}"
            )
        pair_valid_mask &= valid_mask[:, pair_i] & valid_mask[:, pair_j]
    if not pair_valid_mask.any():
        return float("nan")
    correct = torch.sign(pred_diff[pair_valid_mask]) == torch.sign(target_diff[pair_valid_mask])
    return float(correct.float().mean().item())


def _abs_rel_degradation_targets(
    predicted_score: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    canonical_abs_rel: torch.Tensor,
    eps: float = _DEGRADATION_EPS,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build all AbsRel degradation targets with one shared validity mask."""
    predicted_score = predicted_score.detach().float().flatten()
    candidate_abs_rel = candidate_abs_rel.detach().float().flatten()
    canonical_abs_rel = canonical_abs_rel.detach().float().flatten()

    if not (predicted_score.shape == candidate_abs_rel.shape == canonical_abs_rel.shape):
        raise ValueError(
            "predicted_score, candidate_abs_rel, and canonical_abs_rel must have "
            "the same flattened shape"
        )

    # AbsRel is non-negative. Negative values (notably the CSV -1 sentinel)
    # must not participate in correlations or selection.
    valid_mask = (
        torch.isfinite(predicted_score)
        & torch.isfinite(candidate_abs_rel)
        & torch.isfinite(canonical_abs_rel)
        & (candidate_abs_rel >= 0)
        & (canonical_abs_rel >= 0)
    )

    absolute = candidate_abs_rel - canonical_abs_rel
    percentage = absolute / (canonical_abs_rel + eps) * 100.0
    log_ratio = torch.log(candidate_abs_rel + eps) - torch.log(canonical_abs_rel + eps)
    valid_mask &= (
        torch.isfinite(absolute)
        & torch.isfinite(percentage)
        & torch.isfinite(log_ratio)
    )

    return {
        "abs_rel_degradation": absolute,
        "abs_rel_degradation_percent": percentage,
        "abs_rel_degradation_log": log_ratio,
    }, valid_mask


def _degradation_correlation_metrics(
    predicted_score: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    canonical_abs_rel: torch.Tensor,
    max_samples: int,
) -> Dict[str, float]:
    targets, valid_mask = _abs_rel_degradation_targets(
        predicted_score,
        candidate_abs_rel,
        canonical_abs_rel,
    )
    metrics: Dict[str, float] = {}
    for target_name, target in targets.items():
        metrics.update(
            compute_vector_masked_correlations(
                target,
                predicted_score,
                valid_mask=valid_mask,
                prefix=f"q_vs_{target_name}",
                max_samples=max_samples,
            )
        )
    return metrics


def _condition_groupwise_correlation_metrics(
    predicted_score: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    canonical_abs_rel: torch.Tensor,
    scene_id: torch.Tensor,
    motion_id: torch.Tensor,
    light_id: torch.Tensor,
    max_samples: int,
) -> Dict[str, float]:
    """Macro-average correlations over (scene, motion, light) conditions."""
    predicted_score = predicted_score.detach().float().flatten()
    candidate_abs_rel = candidate_abs_rel.detach().float().flatten()
    canonical_abs_rel = canonical_abs_rel.detach().float().flatten()
    conditions = torch.stack(
        [scene_id.flatten(), motion_id.flatten(), light_id.flatten()],
        dim=1,
    )
    if not (
        predicted_score.shape
        == candidate_abs_rel.shape
        == canonical_abs_rel.shape
        == conditions[:, 0].shape
    ):
        raise ValueError("condition metadata must have the same sample count as predicted_score")
    condition_valid = torch.isfinite(conditions).all(dim=1)
    metric_keys = tuple(
        f"q_vs_{target_name}_{correlation_name}"
        for target_name in (
            "abs_rel_degradation",
            "abs_rel_degradation_percent",
            "abs_rel_degradation_log",
        )
        for correlation_name in ("pearson", "spearman")
    )
    correlation_values: dict[str, list[float]] = {key: [] for key in metric_keys}

    for condition in torch.unique(conditions[condition_valid], dim=0):
        condition_mask = condition_valid & (conditions == condition).all(dim=1)
        condition_metrics = _degradation_correlation_metrics(
            predicted_score[condition_mask],
            candidate_abs_rel[condition_mask],
            canonical_abs_rel[condition_mask],
            max_samples,
        )
        for key, value in condition_metrics.items():
            if torch.isfinite(torch.tensor(value)):
                correlation_values[key].append(value)

    return {
        f"condition_groupwise_mean_{key}": (
            float(torch.tensor(values).mean().item()) if values else float("nan")
        )
        for key, values in correlation_values.items()
    }


def _selection_metrics(
    predicted_score: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    canonical_abs_rel: torch.Tensor,
    group_id: torch.Tensor,
    eps: float = _DEGRADATION_EPS,
) -> Dict[str, float]:
    predicted_score = predicted_score.detach().float().flatten()
    candidate_abs_rel = candidate_abs_rel.detach().float().flatten()
    canonical_abs_rel = canonical_abs_rel.detach().float().flatten()
    targets, valid_mask = _abs_rel_degradation_targets(
        predicted_score,
        candidate_abs_rel,
        canonical_abs_rel,
        eps=eps,
    )
    group_id = group_id.detach().flatten()
    if group_id.shape != valid_mask.shape:
        raise ValueError("group_id must have the same flattened shape as predicted_score")
    valid_mask &= torch.isfinite(group_id.float())

    exact_hits: list[float] = []
    relative_regrets: list[float] = []
    near_oracle_hits: dict[float, list[float]] = {
        threshold: [] for threshold in _NEAR_ORACLE_THRESHOLDS
    }

    for current_group_id in torch.unique(group_id[valid_mask]):
        group_mask = valid_mask & (group_id == current_group_id)
        if int(group_mask.sum().item()) < 2:
            continue

        group_q = predicted_score[group_mask]
        group_candidate_abs_rel = candidate_abs_rel[group_mask]
        group_canonical_abs_rel = canonical_abs_rel[group_mask]

        canonical_spread = group_canonical_abs_rel.max() - group_canonical_abs_rel.min()
        canonical_tolerance = eps + 1e-6 * group_canonical_abs_rel.abs().max()
        if canonical_spread > canonical_tolerance:
            raise RuntimeError(
                "canonical AbsRel is inconsistent inside one selection group; "
                "check grouping and canonical matching"
            )

        selected_index = int(torch.argmin(group_q).item())
        transformed_hits = []
        for target in targets.values():
            group_target = target[group_mask]
            transformed_hits.append(
                bool((group_target[selected_index] == group_target.min()).item())
            )
        if len(set(transformed_hits)) != 1:
            raise RuntimeError(
                "absolute/percentage/log-ratio selection accuracy diverged; "
                "check grouping, canonical matching, and filtering"
            )

        selected_abs_rel = group_candidate_abs_rel[selected_index]
        oracle_abs_rel = group_candidate_abs_rel.min()
        relative_regret = (selected_abs_rel - oracle_abs_rel) / (oracle_abs_rel + eps) * 100.0
        relative_regret_value = float(relative_regret.item())

        exact_hits.append(float(transformed_hits[0]))
        relative_regrets.append(relative_regret_value)
        for threshold in _NEAR_ORACLE_THRESHOLDS:
            near_oracle_hits[threshold].append(float(relative_regret_value <= threshold))

    if not exact_hits:
        return {
            "selection_exact_accuracy": float("nan"),
            "selection_relative_regret_percent": float("nan"),
            **{
                f"selection_near_oracle_{int(threshold)}pct_rate": float("nan")
                for threshold in _NEAR_ORACLE_THRESHOLDS
            },
        }

    return {
        "selection_exact_accuracy": sum(exact_hits) / len(exact_hits),
        "selection_relative_regret_percent": sum(relative_regrets) / len(relative_regrets),
        **{
            f"selection_near_oracle_{int(threshold)}pct_rate": sum(hits) / len(hits)
            for threshold, hits in near_oracle_hits.items()
        },
    }


def _new_accumulator() -> dict:
    return {
        "loss": 0.0,
        "nll_loss": 0.0,
        "mean_loss": 0.0,
        "variance_loss": 0.0,
        "ranking_loss": 0.0,
        "processed_batches": 0,
        "q_rank_accuracy": 0.0,
        "q_rank_accuracy_count": 0,
        "vectors": {
            "target_ssi_loss": [],
            "camera_bias": [],
            "sigma": [],
            "q_score": [],
            "candidate_abs_rel": [],
            "canonical_abs_rel": [],
            "abs_rel_degradation": [],
            "rmse_degradation": [],
            "group_id": [],
            "scene_id": [],
            "motion_id": [],
            "light_id": [],
        },
    }


def _append_vectors(
    accumulator: dict,
    sample_mask: torch.Tensor | None = None,
    **items: torch.Tensor,
) -> None:
    for key, value in items.items():
        value = value.detach().float().flatten()
        if sample_mask is not None:
            value = value[sample_mask]
        if value.numel() > 0:
            accumulator["vectors"][key].append(value.cpu())


def _add_rank_accuracy(
    accumulator: dict,
    rank_accuracy: float,
) -> None:
    if torch.isfinite(torch.tensor(rank_accuracy)):
        accumulator["q_rank_accuracy"] += rank_accuracy
        accumulator["q_rank_accuracy_count"] += 1


def _finalize_accumulator(
    accumulator: dict,
    max_samples: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    processed_batches = accumulator["processed_batches"]
    if processed_batches > 0:
        for key in ("loss", "nll_loss", "mean_loss", "variance_loss", "ranking_loss"):
            metrics[key] = accumulator[key] / processed_batches

    rank_count = accumulator["q_rank_accuracy_count"]
    if rank_count > 0:
        metrics["q_rank_accuracy"] = accumulator["q_rank_accuracy"] / rank_count

    cat_vectors = {
        key: torch.cat(values, dim=0) if values else None
        for key, values in accumulator["vectors"].items()
    }
    metadata_keys = {"group_id", "scene_id", "motion_id", "light_id"}
    for key, value in cat_vectors.items():
        if value is not None and key != "rmse_degradation" and key not in metadata_keys:
            metrics[key] = _finite_mean(value)

    target_loss = cat_vectors.get("target_ssi_loss")
    camera_bias = cat_vectors.get("camera_bias")
    abs_rel_degradation = cat_vectors.get("abs_rel_degradation")
    rmse_degradation = cat_vectors.get("rmse_degradation")
    q_score = cat_vectors.get("q_score")

    if target_loss is not None and camera_bias is not None:
        metrics.update(
            compute_vector_masked_correlations(
                target_loss,
                camera_bias,
                prefix="bias_vs_ssi_loss",
                max_samples=max_samples,
            )
        )
    if abs_rel_degradation is not None and camera_bias is not None:
        metrics.update(
            compute_vector_masked_correlations(
                abs_rel_degradation,
                camera_bias,
                prefix="bias_vs_abs_rel_degradation",
                max_samples=max_samples,
            )
        )
    candidate_abs_rel = cat_vectors.get("candidate_abs_rel")
    canonical_abs_rel = cat_vectors.get("canonical_abs_rel")
    group_id = cat_vectors.get("group_id")
    scene_id = cat_vectors.get("scene_id")
    motion_id = cat_vectors.get("motion_id")
    light_id = cat_vectors.get("light_id")
    if q_score is not None and candidate_abs_rel is not None and canonical_abs_rel is not None:
        metrics.update(
            _degradation_correlation_metrics(
                q_score,
                candidate_abs_rel,
                canonical_abs_rel,
                max_samples,
            )
        )
        if group_id is not None:
            metrics.update(
                _selection_metrics(
                    q_score,
                    candidate_abs_rel,
                    canonical_abs_rel,
                    group_id,
                )
            )
        if scene_id is not None and motion_id is not None and light_id is not None:
            metrics.update(
                _condition_groupwise_correlation_metrics(
                    q_score,
                    candidate_abs_rel,
                    canonical_abs_rel,
                    scene_id,
                    motion_id,
                    light_id,
                    max_samples,
                )
            )
    if rmse_degradation is not None and q_score is not None:
        metrics.update(
            compute_vector_masked_correlations(
                rmse_degradation,
                q_score,
                prefix="q_vs_rmse_degradation",
                max_samples=max_samples,
            )
        )
    return metrics


@torch.no_grad()
def validate(
    epoch: int,
    model_id: str,
    model,
    loader: DataLoader,
    device,
    amp: bool,
    lambda_smooth_logvar: float,
    lambda_variance: float,
    listnet_temperature: float,
    uncertainty_mode: str,
    list_loss_weight: float,
    seen_topology_numbers: torch.Tensor = None,
    unseen_topology_numbers: torch.Tensor = None,
    correlation_max_samples: int = 100_000,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    relative_align_mode: str = "scale_shift",
    uncertainty_alpha: float = 1.0,
):
    del model_id, lambda_smooth_logvar, uncertainty_mode, min_depth, max_depth, relative_align_mode

    model.eval()
    total_accumulator = _new_accumulator()
    seen_accumulator = _new_accumulator()
    unseen_accumulator = _new_accumulator()
    scene_ids: dict[str, int] = {}

    progress_bar = tqdm(
        loader,
        desc=f"Validation {epoch:03d}",
        dynamic_ncols=True,
        leave=False,
    )

    for step, batch in enumerate(progress_bar, start=1):
        if batch is None:
            continue

        num_groups, num_candidates = batch["candidate_images"].shape[:2]
        flat_batch = tensor_device(flatten_group_batch(batch), device)
        candidate_imgs = flat_batch["candidate_images"]
        canonical_imgs = flat_batch["canonical_images"]
        camera_context = flat_batch["camera_context"]
        abs_rel_degradation = flat_batch["abs_rel_degradation"]
        rmse_degradation = flat_batch["rmse_degradation"]

        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(
                candidate_imgs,
                canonical_imgs,
                camera_context,
                target_size=candidate_imgs.shape[-2:],
            )
            # target_loss = scale_shift_invariant_depth_loss(
            #     out["candidate_depth"],
            #     out["canonical_depth"],
            # )
            target_loss = log_scale_invariant_depth_difference(
                out["candidate_depth"],
                out["canonical_depth"],
            )
            mean_loss, variance_loss = scalar_heteroscedastic_laplace_loss( # scalar_heteroscedastic_loss(
                out["camera_bias"],
                out["variance"],
                target_loss,
            )
            q_score = out["camera_bias"] # + uncertainty_alpha * out["std"]
            group_q = reshape_group_batch(q_score, num_groups, num_candidates)
            group_degradation = reshape_group_batch(
                abs_rel_degradation,
                num_groups,
                num_candidates,
            )
            ranking_loss = signed_pairwise_ranknet_loss(
                group_q,
                group_degradation,
                temperature=listnet_temperature,
            )
            nll_loss = mean_loss + lambda_variance * variance_loss
            loss = nll_loss + list_loss_weight * ranking_loss

        batch_vectors = {
            "target_ssi_loss": target_loss,
            "camera_bias": out["camera_bias"],
            "sigma": out["std"],
            "q_score": q_score,
            "candidate_abs_rel": flat_batch["candidate_abs_rel"],
            "canonical_abs_rel": flat_batch["canonical_abs_rel"],
            "abs_rel_degradation": abs_rel_degradation,
            "rmse_degradation": rmse_degradation,
        }
        group_candidate_abs_rel = reshape_group_batch(
            flat_batch["candidate_abs_rel"],
            num_groups,
            num_candidates,
        )
        group_canonical_abs_rel = reshape_group_batch(
            flat_batch["canonical_abs_rel"],
            num_groups,
            num_candidates,
        )
        group_valid_mask = (
            torch.isfinite(group_q)
            & torch.isfinite(group_candidate_abs_rel)
            & torch.isfinite(group_canonical_abs_rel)
            & (group_candidate_abs_rel >= 0)
            & (group_canonical_abs_rel >= 0)
        )
        evaluation_group_degradation = group_candidate_abs_rel - group_canonical_abs_rel
        rank_accuracy = _pairwise_rank_accuracy(
            group_q,
            evaluation_group_degradation,
            valid_mask=group_valid_mask,
        )

        group_info = batch["info"].to(device=device)
        batch_scene_ids = torch.tensor(
            [scene_ids.setdefault(str(scene), len(scene_ids)) for scene in batch["scene"]],
            device=device,
            dtype=torch.float32,
        )
        batch_vectors.update(
            {
                "group_id": batch["group_index"].to(device=device)[:, None]
                .expand(-1, num_candidates)
                .reshape(-1),
                "scene_id": batch_scene_ids[:, None].expand(-1, num_candidates).reshape(-1),
                "motion_id": group_info[:, 3:4].expand(-1, num_candidates).reshape(-1),
                "light_id": group_info[:, 2:3].expand(-1, num_candidates).reshape(-1),
            }
        )

        total_accumulator["loss"] += float(loss.item())
        total_accumulator["nll_loss"] += float(nll_loss.item())
        total_accumulator["mean_loss"] += float(mean_loss.item())
        total_accumulator["variance_loss"] += float(variance_loss.item())
        total_accumulator["ranking_loss"] += float(ranking_loss.item())
        total_accumulator["processed_batches"] += 1
        _add_rank_accuracy(total_accumulator, rank_accuracy)
        _append_vectors(total_accumulator, **batch_vectors)

        group_topology = group_info[:, 6].long()
        if seen_topology_numbers is not None:
            seen_group_mask = torch.isin(
                group_topology,
                seen_topology_numbers.to(device=device).long(),
            )
            seen_sample_mask = seen_group_mask[:, None].expand(-1, num_candidates).reshape(-1)
            _append_vectors(seen_accumulator, seen_sample_mask, **batch_vectors)
            if seen_group_mask.any():
                _add_rank_accuracy(
                    seen_accumulator,
                    _pairwise_rank_accuracy(
                        group_q[seen_group_mask],
                        group_degradation[seen_group_mask],
                    ),
                )
        if unseen_topology_numbers is not None:
            unseen_group_mask = torch.isin(
                group_topology,
                unseen_topology_numbers.to(device=device).long(),
            )
            unseen_sample_mask = unseen_group_mask[:, None].expand(-1, num_candidates).reshape(-1)
            _append_vectors(unseen_accumulator, unseen_sample_mask, **batch_vectors)
            if unseen_group_mask.any():
                _add_rank_accuracy(
                    unseen_accumulator,
                    _pairwise_rank_accuracy(
                        group_q[unseen_group_mask],
                        group_degradation[unseen_group_mask],
                    ),
                )

        n = max(total_accumulator["processed_batches"], 1)
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_accumulator['loss'] / n:.4f}",
            ssi=f"{_finite_mean(target_loss):.4f}",
            deg=f"{_finite_mean(abs_rel_degradation):.4f}",
            q_acc=f"{total_accumulator['q_rank_accuracy'] / max(total_accumulator['q_rank_accuracy_count'], 1):.4f}",
        )

    total_metrics = _finalize_accumulator(total_accumulator, correlation_max_samples)
    seen_metrics = _finalize_accumulator(seen_accumulator, correlation_max_samples)
    unseen_metrics = _finalize_accumulator(unseen_accumulator, correlation_max_samples)

    return total_metrics, seen_metrics, unseen_metrics
