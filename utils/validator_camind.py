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
from model.loss_fn import (
    scalar_heteroscedastic_loss,
    signed_pairwise_ranknet_loss,
)
from model.loss_target import ssi_independent_meter_space_depth_loss
from utils.train_utils import reshape_group_batch, tensor_device


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
    selection_min_settings: int = 10,
    selection_thresholds: Sequence[
        float
    ] = DEFAULT_RELATIVE_REGRET_THRESHOLDS,
    selection_alpha_values: Sequence[float] = (0.0, 0.5, 1.0),
):
    del model_id, lambda_smooth_logvar, uncertainty_mode, min_depth, max_depth, relative_align_mode

    loader.dataset.load_depth = True
    model.eval()
    total_accumulator = new_validation_accumulator()
    seen_accumulator = new_validation_accumulator()
    unseen_accumulator = new_validation_accumulator()

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
                camera_context,
                target_size=candidate_imgs.shape[-2:],
            )
            target_loss = ssi_independent_meter_space_depth_loss(
                out["candidate_depth"],
                out["canonical_depth"],
                candidate_gt_depth,
                canonical_gt_depth,
            )
            mean_loss, variance_loss = scalar_heteroscedastic_loss(
                out["camera_bias"],
                out["variance"],
                target_loss,
            )
            q_score = out["camera_bias"] + uncertainty_alpha * out["std"]
            group_q = reshape_group_batch(q_score, num_groups, num_candidates)
            group_target_loss = reshape_group_batch(
                target_loss,
                num_groups,
                num_candidates,
            )
            ranking_loss = signed_pairwise_ranknet_loss(
                group_q,
                group_target_loss,
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
