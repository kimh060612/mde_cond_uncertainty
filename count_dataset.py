from pathlib import Path

import hydra
from omegaconf import DictConfig

from dataset.ati_dataset_caminduce import CameraParameterRange, FoundationCameraGroupedDataset


@hydra.main(version_base=None, config_path="config", config_name="base_caminduce")
def main(cfg: DictConfig) -> None:
    csv_root = Path(cfg.dataset.csv_path)
    csv_paths = [csv_root] if csv_root.is_file() else sorted(csv_root.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in {csv_root}")

    group_size = int(cfg.training.group_size)
    dataset_kwargs = {
        "csv_paths": csv_paths,
        "foundation_model_name": cfg.model.model_id,
        "camera_model_name": cfg.model.camera_model_name,
        "parameter_range": CameraParameterRange(
            exposure_min=cfg.dataset.exposure_min,
            exposure_max=cfg.dataset.exposure_max,
            gain_min=cfg.dataset.gain_min,
            gain_max=cfg.dataset.gain_max,
        ),
        "candidates_per_group": group_size,
        "min_overlap_ratio": cfg.dataset.min_registration_overlap_ratio,
        "min_ecc_score": cfg.dataset.min_registration_ecc_score,
        "max_time_diff_sec": cfg.dataset.max_pair_time_diff_sec,
        "max_registration_translation_px": cfg.dataset.max_registration_translation_px,
        "include_canonical_setting_as_candidate": cfg.dataset.include_canonical_setting_as_candidate,
        "load_images": False,
    }
    seen_set = FoundationCameraGroupedDataset(
        **dataset_kwargs,
        topologies=cfg.dataset.seen_val_topologies,
    )
    unseen_set = FoundationCameraGroupedDataset(
        **dataset_kwargs,
        topologies=cfg.dataset.unseen_val_topologies,
    )

    print(f"Seen Dataset: total groups={len(seen_set):,}, total frames={len(seen_set) * group_size:,}")
    print(f"Unseen Dataset: total groups={len(unseen_set):,}, total frames={len(unseen_set) * group_size:,}")


if __name__ == "__main__":
    main()
