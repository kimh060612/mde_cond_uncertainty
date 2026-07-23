from __future__ import annotations

import csv
from glob import glob
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

from dataset.ati_dataset_caminduce import (
    CameraParameterRange,
    FoundationCameraGroupedDataset,
    PairedResizeToTensor,
)
from evaluation_utils.eval_selection import (
    compute_selection_alpha_sweep,
    plot_selection_alpha_sweep,
)
from model.dav2_ati_model import MODEL_IDS
from model.dav2_camerror_model import CameraInducedErrorModel
from utils.train_utils import seed_everything, topology_id


def resolve_checkpoint_path(cfg: DictConfig) -> Path:
    configured_path = cfg.evaluation.get("checkpoint_path")
    if configured_path:
        checkpoint_path = Path(to_absolute_path(str(configured_path)))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        return checkpoint_path

    checkpoint_dir = Path(to_absolute_path(str(cfg.dataset.output_dir)))
    candidates = list(checkpoint_dir.glob("ckpt_model_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint was configured and none was found under "
            f"{checkpoint_dir}. Set evaluation.checkpoint_path with Hydra."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_validation_dataset(
    cfg: DictConfig,
    image_processor: AutoImageProcessor,
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
        seed=cfg.training.seed,
    )


@torch.no_grad()
def collect_predictions(
    model: CameraInducedErrorModel,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    inference_batch_size: int,
) -> dict[str, torch.Tensor]:
    collected = {
        "camera_bias": [],
        "camera_std": [],
        "candidate_abs_rel": [],
        "group_id": [],
        "topology": [],
    }
    model.eval()
    for batch in tqdm(
        loader,
        desc="Selection inference",
        dynamic_ncols=True,
    ):
        candidate_images = batch["candidate_images"].squeeze(0)
        camera_context = batch["camera_context"].squeeze(0)
        group_size = candidate_images.shape[0]

        bias_chunks = []
        std_chunks = []
        for start in range(0, group_size, inference_batch_size):
            stop = start + inference_batch_size
            images = candidate_images[start:stop].to(
                device=device,
                non_blocking=True,
            )
            context = camera_context[start:stop].to(
                device=device,
                non_blocking=True,
            )
            with torch.autocast(
                device_type=device.type,
                enabled=amp,
            ):
                output = model.inference(
                    images,
                    context,
                    target_size=images.shape[-2:],
                )
            bias_chunks.append(output["camera_bias"].float().cpu())
            std_chunks.append(output["std"].float().cpu())

        collected["camera_bias"].append(torch.cat(bias_chunks))
        collected["camera_std"].append(torch.cat(std_chunks))
        collected["candidate_abs_rel"].append(
            batch["candidate_abs_rel"].flatten().float()
        )
        collected["group_id"].append(
            batch["group_index"].flatten().repeat_interleave(group_size)
        )
        collected["topology"].append(
            batch["info"][:, 6].flatten().repeat_interleave(group_size)
        )

    return {
        key: torch.cat(values).flatten()
        for key, values in collected.items()
    }


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="base_caminduce",
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not cfg.training.no_amp
    checkpoint_path = resolve_checkpoint_path(cfg)
    model_id = MODEL_IDS[cfg.model.model_id]

    processor_source = (
        str(checkpoint_path.parent)
        if (checkpoint_path.parent / "preprocessor_config.json").is_file()
        else model_id
    )
    image_processor = AutoImageProcessor.from_pretrained(
        processor_source,
        use_fast=False,
    )
    dataset = build_validation_dataset(cfg, image_processor)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CameraInducedErrorModel(
        model_id=model_id,
        context_dim=dataset.condition_dim,
        cache_dir=None,
        feature_channels=cfg.model.uncertainty_width,
        hidden_channels=cfg.model.uncertainty_width,
        film_hidden_dim=cfg.model.film_layer_width,
        max_bias=cfg.training.max_bias,
        min_log_variance=cfg.training.min_log_var,
        max_log_variance=cfg.training.max_log_var,
        initial_std=cfg.training.initial_std,
        variance_head_init_std=cfg.training.variance_head_init_std,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    predictions = collect_predictions(
        model,
        loader,
        device,
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
        )
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

    figure = plot_selection_alpha_sweep(
        {
            split_name: sweeps[split_name]
            for split_name in ("all", "seen", "unseen")
        }
    )
    curve_path = output_dir / "selection_alpha_sweep.png"
    figure.savefig(curve_path, dpi=200)
    import matplotlib.pyplot as plt
    plt.close(figure)

    print(f"checkpoint: {checkpoint_path}")
    print(f"validation groups: {len(dataset):,}")
    print(f"selection CSV: {csv_path}")
    print(f"alpha-sweep curve: {curve_path}")


if __name__ == "__main__":
    main()
