import csv
import math
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dataset.ae_dataset import AutoExposureMotionDataset, natural_key
from finetune.utils import (
    align_affine_depth,
    depth_metrics,
    metric3d_input,
    remove_padding,
    seed_everything,
)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return index, self.dataset[index]


def predict_metric3d(
    model,
    rgb: torch.Tensor,
    focal_length: float,
) -> torch.Tensor:
    model_input, padding, resize_scale = metric3d_input(rgb)
    prediction, _, _ = model.inference({"input": model_input})
    prediction = remove_padding(prediction, padding, rgb.shape[-2:]).squeeze(1)
    return prediction * (focal_length * resize_scale / 1000.0)


@torch.inference_mode()
def evaluate(
    model,
    dataset: AutoExposureMotionDataset,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> list[dict[str, object]]:
    totals = defaultdict(lambda: [0.0, 0.0, 0, 0])
    modes = ("metric", "affine_invariant")

    for indices, sample in tqdm(loader, desc="metric3d_v2", dynamic_ncols=True):
        rgb, target, valid, _, _ = sample
        rgb = rgb.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        valid = valid.to(device, dtype=torch.bool, non_blocking=True)

        with torch.autocast(device.type, enabled=device.type == "cuda"):
            prediction = predict_metric3d(
                model, rgb, cfg.evaluation.metric3d_focal_length
            )

        for mode in modes:
            evaluated = (
                align_affine_depth(prediction, target, valid)
                if mode == "affine_invariant"
                else prediction
            )
            abs_rel, a1 = depth_metrics(
                evaluated,
                target,
                valid,
                cfg.dataset.min_depth,
                cfg.dataset.max_depth,
            )
            valid_ratios = valid.float().flatten(1).mean(1)

            for offset, dataset_index in enumerate(indices.tolist()):
                item = dataset.items[dataset_index]
                key = (item.scene_name, item.lap_id, mode)
                frame_abs_rel = float(abs_rel[offset])
                frame_a1 = float(a1[offset])
                if (
                    valid_ratios[offset] < cfg.dataset.min_valid_depth_ratio
                    or not math.isfinite(frame_abs_rel)
                    or not math.isfinite(frame_a1)
                ):
                    totals[key][3] += 1
                    continue
                totals[key][0] += frame_abs_rel
                totals[key][1] += frame_a1
                totals[key][2] += 1

    rows = []
    for (scene_name, lap_id, mode), (abs_rel, a1, count, skipped) in sorted(
        totals.items(), key=lambda pair: natural_key(pair[0])
    ):
        item = next(
            item
            for item in dataset.items
            if item.scene_name == scene_name and item.lap_id == lap_id
        )
        rows.append(
            {
                "scene_name": scene_name,
                "lap_id": lap_id,
                "light": item.light,
                "collection_speed": item.collection_speed,
                "topology": item.topology,
                "model_name": "metric3d_v2",
                "model_id": (
                    f"{cfg.evaluation.metric3d_model_id}:"
                    f"{cfg.evaluation.metric3d_model}"
                ),
                "depth_type": mode,
                "alignment": "scale_shift" if mode == "affine_invariant" else "none",
                "evaluated_frames": count,
                "skipped_frames": skipped,
                "abs_rel": abs_rel / count if count else float("nan"),
                "a1": a1 / count if count else float("nan"),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(config_path="../config", config_name="base_finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AutoExposureMotionDataset(
        root_dir=cfg.dataset.dataset_root,
        image_processor=None,
        image_size=(cfg.model.image_height, cfg.model.image_width),
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        min_valid_depth_ratio=cfg.dataset.min_valid_depth_ratio,
        min_length=1,
    )
    if not dataset.sequences:
        raise FileNotFoundError(f"No auto-exposure laps found in {cfg.dataset.dataset_root}")
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=device.type == "cuda",
    )

    # The official hub entry downloads its checkpoint from JUGGHM/Metric3D on HF.
    metric3d = torch.hub.load(
        "yvanyin/metric3d",
        cfg.evaluation.metric3d_model,
        pretrain=True,
        trust_repo=True,
    ).to(device).eval()
    rows = evaluate(metric3d, dataset, loader, device, cfg)

    output_path = Path(cfg.evaluation.output_csv).resolve()
    write_csv(rows, output_path)
    print(f"saved {len(rows)} rows to {output_path}")


def test_affine_depth_evaluation() -> None:
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    prediction = (target - 1.0) / 2.0
    valid = torch.ones_like(target, dtype=torch.bool)
    aligned = align_affine_depth(prediction, target, valid)
    abs_rel, a1 = depth_metrics(aligned, target, valid, 1e-3, 10.0)
    assert torch.allclose(aligned, target, atol=1e-5)
    assert torch.allclose(abs_rel, torch.zeros_like(abs_rel), atol=1e-5)
    assert torch.equal(a1, torch.ones_like(a1))


if __name__ == "__main__":
    main()
