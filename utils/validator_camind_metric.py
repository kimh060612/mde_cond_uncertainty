from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.ati_dataset_caminduce import flatten_group_batch
from evaluation_utils.eval_selection import (
    DEFAULT_RELATIVE_REGRET_THRESHOLDS,
    compute_selection_alpha_sweep,
)
from evaluation_utils.eval_utils import (
    add_rank_counts,
    append_accumulator_vectors,
    concatenate_accumulator_vectors,
    finalize_validation_accumulator,
    finite_mean,
    new_validation_accumulator,
    pairwise_rank_counts,
)
# from transformers import AutoModelForDepthEstimation
# from model.dav2_camerror_model import forward_with_rgb_model, inference_with_rgb_model
from model.loss_fn import (
    groupwise_soft_optimal_loss,
    groupwise_pairwise_probit_loss,
    scalar_heteroscedastic_loss,
    signed_pairwise_ranknet_loss,
)
from model.loss_target import metric_meter_space_depth_loss
from utils.train_utils import reshape_group_batch, tensor_device


def _update_optimal_selection_stats(
    stats: dict[str, float],
    group_bias: torch.Tensor,
    group_degradation: torch.Tensor,
    target_temperature: float,
    prediction_temperature: float,
    group_mask: torch.Tensor | None = None,
) -> None:
    if group_mask is not None:
        group_bias = group_bias[group_mask]
        group_degradation = group_degradation[group_mask]
    if group_bias.shape[0] == 0:
        return

    optimal_candidate_index = group_degradation.argmin(dim=1)
    selected_index = group_bias.argmin(dim=1)
    selected_degradation = group_degradation.gather(
        1, selected_index[:, None]
    ).squeeze(1)
    optimal_degradation = group_degradation.gather(
        1, optimal_candidate_index[:, None]
    ).squeeze(1)
    num_groups = group_bias.shape[0]
    stats["loss_sum"] += float(
        groupwise_soft_optimal_loss(
            group_bias,
            group_degradation,
            target_temperature,
            prediction_temperature,
        ).item()
    ) * num_groups
    stats["correct"] += int((selected_index == optimal_candidate_index).sum().item())
    stats["regret_sum"] += float(
        (selected_degradation - optimal_degradation).clamp_min(0).sum().item()
    )
    stats["num_groups"] += num_groups


def pairwise_policy_roc_data(
    camera_mean: torch.Tensor,
    camera_std: torch.Tensor,
    candidate_abs_rel: torch.Tensor,
    group_id: torch.Tensor,
    tie_eps_percents: Sequence[float],
    m_switch: float,
    beta: float,
    eps: float = 1e-6,
) -> dict[float, dict[str, object]]:
    """Evaluate both directions of every unordered pair, separately per group."""
    z_parts, label_parts, gap_parts = [], [], []
    for value in torch.unique(group_id):
        mask = group_id == value
        mean = camera_mean[mask].float()
        std = camera_std[mask].float()
        actual = candidate_abs_rel[mask].float()
        if mean.numel() < 2:
            continue
        pair_i, pair_j = torch.triu_indices(mean.numel(), mean.numel(), offset=1)
        valid = (
            torch.isfinite(mean[pair_i])
            & torch.isfinite(mean[pair_j])
            & torch.isfinite(std[pair_i])
            & torch.isfinite(std[pair_j])
            & torch.isfinite(actual[pair_i])
            & torch.isfinite(actual[pair_j])
            & (std[pair_i] >= 0)
            & (std[pair_j] >= 0)
            & (actual[pair_i] >= 0)
            & (actual[pair_j] >= 0)
        )
        pair_i, pair_j = pair_i[valid], pair_j[valid]
        if pair_i.numel() == 0:
            continue
        pair_std = torch.hypot(std[pair_i], std[pair_j])
        gap = (actual[pair_i] - actual[pair_j]).abs() / (
            torch.minimum(actual[pair_i], actual[pair_j]) + eps
        ) * 100.0
        z_parts.append(torch.cat([
            (mean[pair_i] - mean[pair_j] - m_switch) / (pair_std + eps),
            (mean[pair_j] - mean[pair_i] - m_switch) / (pair_std + eps),
        ]))
        label_parts.append(torch.cat([
            actual[pair_j] < actual[pair_i],
            actual[pair_i] < actual[pair_j],
        ]))
        gap_parts.append(torch.cat([gap, gap]))

    result: dict[float, dict[str, object]] = {}
    if not z_parts:
        return result
    z_all = torch.cat(z_parts)
    label_all = torch.cat(label_parts)
    gap_all = torch.cat(gap_parts)
    for tie_eps in tie_eps_percents:
        keep = gap_all >= float(tie_eps)
        z, label = z_all[keep], label_all[keep]
        positives = int(label.sum().item())
        negatives = int((~label).sum().item())
        if positives == 0 or negatives == 0:
            continue
        order = torch.argsort(z, descending=True)
        sorted_z, sorted_label = z[order], label[order]
        ends = torch.cat([
            torch.nonzero(sorted_z[:-1] != sorted_z[1:], as_tuple=False).flatten(),
            sorted_z.new_tensor([sorted_z.numel() - 1], dtype=torch.long),
        ])
        true_positive = sorted_label.long().cumsum(0)[ends].float()
        false_positive = (~sorted_label).long().cumsum(0)[ends].float()
        tpr = torch.cat([true_positive.new_zeros(1), true_positive / positives])
        fpr = torch.cat([false_positive.new_zeros(1), false_positive / negatives])
        decision = z > beta
        tp = int((decision & label).sum().item())
        fp = int((decision & ~label).sum().item())
        result[float(tie_eps)] = {
            "fpr": fpr.cpu(),
            "tpr": tpr.cpu(),
            "auroc": float(torch.trapz(tpr, fpr).item()),
            "accuracy": float((decision == label).float().mean().item()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / positives,
            "operating_fpr": fp / negatives,
            "num_pairs": int(label.numel()),
            "num_switches": int(decision.sum().item()),
            "scores": z.cpu(),
            "labels": label.cpu(),
            "decisions": decision.cpu(),
        }
    return result


def plot_pairwise_policy_roc(
    vectors: dict[str, torch.Tensor | None],
    tie_eps_percents: Sequence[float],
    m_switch: float,
    beta: float,
):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    required = ("camera_bias", "sigma", "candidate_abs_rel", "group_id")
    if all(vectors.get(key) is not None for key in required):
        curves = pairwise_policy_roc_data(
            *(vectors[key] for key in required),
            tie_eps_percents=tie_eps_percents,
            m_switch=m_switch,
            beta=beta,
        )
        for tie_eps, curve in curves.items():
            label = (
                f"tie<{tie_eps:g}% AUROC={curve['auroc']:.3f} | "
                f"beta={beta:g} op acc={curve['accuracy']:.3f}, prec={curve['precision']:.3f}, "
                f"rec={curve['recall']:.3f}"
            )
            line, = axis.plot(curve["fpr"], curve["tpr"], label=label)
            axis.scatter(
                curve["operating_fpr"], curve["recall"],
                color=line.get_color(), marker="o", zorder=3,
            )
    axis.plot([0, 1], [0, 1], "k--", alpha=0.4)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Pairwise switch policy ROC")
    axis.grid(alpha=0.25)
    if axis.lines and len(axis.lines) > 1:
        axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


@torch.no_grad()
def validate_metric(
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
    use_ranking_loss: bool,
    group_size: int,
    use_soft_optimal_loss: bool,
    soft_optimal_loss_weight: float,
    target_softmax_temperature: float,
    prediction_softmax_temperature: float,
    seen_topology_numbers: torch.Tensor = None,
    unseen_topology_numbers: torch.Tensor = None,
    context_offset: int = 0,
    correlation_max_samples: int = 100_000,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    relative_align_mode: str = "scale_shift",
    uncertainty_alpha: float = 1.0,
    selection_min_settings: int = 10,
    selection_thresholds: Sequence[
        float
    ] = DEFAULT_RELATIVE_REGRET_THRESHOLDS,
    selection_alpha_values: Sequence[float] = (0.0, 0.5, 1.0),
):
    del model_id, lambda_smooth_logvar, uncertainty_mode, relative_align_mode

    loader.dataset.load_depth = True
    model.eval()
    total_accumulator = new_validation_accumulator()
    seen_accumulator = new_validation_accumulator()
    unseen_accumulator = new_validation_accumulator()
    optimal_stats = {
        name: {"loss_sum": 0.0, "correct": 0, "regret_sum": 0.0, "num_groups": 0}
        for name in ("all", "seen", "unseen")
    }

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
        if num_candidates != group_size:
            raise ValueError(
                f"Expected group_size={group_size}, got {num_candidates} candidates"
            )
        flat_batch = tensor_device(flatten_group_batch(batch), device)
        candidate_imgs = flat_batch["candidate_images"]
        canonical_imgs = flat_batch["canonical_images"]
        camera_context = flat_batch["camera_context"] # B X 10
        abs_rel_degradation = flat_batch["abs_rel_degradation"]
        group_degradation = reshape_group_batch(
            abs_rel_degradation, num_groups, num_candidates
        )
        rmse_degradation = flat_batch["rmse_degradation"]
        candidate_gt_depth = F.interpolate(
            flat_batch["candidate_depths"].unsqueeze(1),
            size=candidate_imgs.shape[-2:],
            mode="nearest",
        )
        canonical_gt_depth = F.interpolate(
            flat_batch["canonical_depths"].unsqueeze(1),
            size=canonical_imgs.shape[-2:],
            mode="nearest",
        )

        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(
                candidate_imgs,
                canonical_imgs,
                camera_context[..., context_offset:],
                target_size=candidate_imgs.shape[-2:],
            )
            target_loss = metric_meter_space_depth_loss(
                out["candidate_depth"],
                out["canonical_depth"],
                min_depth=min_depth,
                max_depth=max_depth
            )
            mean_loss, variance_loss = scalar_heteroscedastic_loss(
                out["camera_bias"],
                out["variance"],
                target_loss,
            )
            q_score = out["camera_bias"] + uncertainty_alpha * out["std"]
            group_bias = reshape_group_batch(
                out["camera_bias"], num_groups, num_candidates
            )
            soft_optimal_loss = groupwise_soft_optimal_loss(
                group_bias,
                group_degradation,
                target_softmax_temperature,
                prediction_softmax_temperature,
            )
            group_q = reshape_group_batch(q_score, num_groups, num_candidates)
            group_target_loss = reshape_group_batch(
                target_loss,
                num_groups,
                num_candidates,
            )
            ranking_loss = (
                signed_pairwise_ranknet_loss(
                    group_q,
                    group_target_loss,
                    temperature=listnet_temperature,
                )
                if use_ranking_loss
                else group_bias.new_zeros(())
            )
            nll_loss = mean_loss + lambda_variance * variance_loss
            loss = (
                nll_loss
                + (
                    soft_optimal_loss_weight * soft_optimal_loss
                    if use_soft_optimal_loss
                    else 0.0
                )
                + list_loss_weight * ranking_loss
            )

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
        _update_optimal_selection_stats(
            optimal_stats["all"],
            group_bias,
            group_degradation,
            target_softmax_temperature,
            prediction_softmax_temperature,
        )
        rank_counts = pairwise_rank_counts(
            group_q,
            evaluation_group_degradation,
            valid_mask=group_valid_mask,
        )

        group_info = batch["info"].to(device=device)
        batch_vectors.update(
            {
                "group_id": batch["group_index"].to(device=device)[:, None]
                .expand(-1, num_candidates)
                .reshape(-1),
            }
        )

        total_accumulator["loss"] += float(loss.item())
        total_accumulator["nll_loss"] += float(nll_loss.item())
        total_accumulator["mean_loss"] += float(mean_loss.item())
        total_accumulator["variance_loss"] += float(variance_loss.item())
        total_accumulator["ranking_loss"] += float(ranking_loss.item())
        total_accumulator["processed_batches"] += 1
        add_rank_counts(total_accumulator, rank_counts)
        append_accumulator_vectors(total_accumulator, **batch_vectors)

        group_topology = group_info[:, 6].long()
        if seen_topology_numbers is not None:
            seen_group_mask = torch.isin(
                group_topology,
                seen_topology_numbers.to(device=device).long(),
            )
            seen_sample_mask = seen_group_mask[:, None].expand(-1, num_candidates).reshape(-1)
            append_accumulator_vectors(
                seen_accumulator,
                seen_sample_mask,
                **batch_vectors,
            )
            if seen_group_mask.any():
                _update_optimal_selection_stats(
                    optimal_stats["seen"],
                    group_bias,
                    group_degradation,
                    target_softmax_temperature,
                    prediction_softmax_temperature,
                    seen_group_mask,
                )
                add_rank_counts(
                    seen_accumulator,
                    pairwise_rank_counts(
                        group_q[seen_group_mask],
                        evaluation_group_degradation[seen_group_mask],
                        valid_mask=group_valid_mask[seen_group_mask],
                    ),
                )
        if unseen_topology_numbers is not None:
            unseen_group_mask = torch.isin(
                group_topology,
                unseen_topology_numbers.to(device=device).long(),
            )
            unseen_sample_mask = unseen_group_mask[:, None].expand(-1, num_candidates).reshape(-1)
            append_accumulator_vectors(
                unseen_accumulator,
                unseen_sample_mask,
                **batch_vectors,
            )
            if unseen_group_mask.any():
                _update_optimal_selection_stats(
                    optimal_stats["unseen"],
                    group_bias,
                    group_degradation,
                    target_softmax_temperature,
                    prediction_softmax_temperature,
                    unseen_group_mask,
                )
                add_rank_counts(
                    unseen_accumulator,
                    pairwise_rank_counts(
                        group_q[unseen_group_mask],
                        evaluation_group_degradation[unseen_group_mask],
                        valid_mask=group_valid_mask[unseen_group_mask],
                    ),
                )

        n = max(total_accumulator["processed_batches"], 1)
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_accumulator['loss'] / n:.4f}",
            ssi=f"{finite_mean(target_loss):.4f}",
            deg=f"{finite_mean(abs_rel_degradation):.4f}",
            q_acc=f"{total_accumulator['q_rank_correct'] / max(total_accumulator['q_rank_total'], 1):.4f}",
        )

    accumulators = {
        "all": total_accumulator,
        "seen": seen_accumulator,
        "unseen": unseen_accumulator,
    }
    vectors = {
        name: concatenate_accumulator_vectors(accumulator)
        for name, accumulator in accumulators.items()
    }
    finalized_metrics = {
        name: finalize_validation_accumulator(
            accumulator,
            correlation_max_samples,
            selection_min_settings=selection_min_settings,
            selection_thresholds=selection_thresholds,
            concatenated_vectors=vectors[name],
        )
        for name, accumulator in accumulators.items()
    }
    for name, stats in optimal_stats.items():
        num_groups = stats["num_groups"]
        finalized_metrics[name].update(
            {
                "groupwise_top1_selection_accuracy": (
                    stats["correct"] / num_groups if num_groups else float("nan")
                ),
                "soft_optimal_loss": (
                    stats["loss_sum"] / num_groups if num_groups else float("nan")
                ),
                "mean_selected_regret": (
                    stats["regret_sum"] / num_groups if num_groups else float("nan")
                ),
            }
        )
    selection_sweeps = {}
    for name, split_vectors in vectors.items():
        camera_bias = split_vectors["camera_bias"]
        camera_std = split_vectors["sigma"]
        candidate_abs_rel = split_vectors["candidate_abs_rel"]
        group_id = split_vectors["group_id"]
        if (
            camera_bias is None
            or camera_std is None
            or candidate_abs_rel is None
            or group_id is None
        ):
            selection_sweeps[name] = []
            continue
        selection_sweeps[name] = compute_selection_alpha_sweep(
            camera_bias,
            camera_std,
            candidate_abs_rel,
            group_id,
            selection_alpha_values,
            min_settings_per_group=selection_min_settings,
            relative_regret_thresholds=selection_thresholds,
        )

    return (
        finalized_metrics["all"],
        finalized_metrics["seen"],
        finalized_metrics["unseen"],
        selection_sweeps,
    )
