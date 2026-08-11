import random

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None, scaler=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "abs_rel": 0.0, "a1": 0.0, "pixels": 0}

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            prediction = model(pixel_values=pixel_values).predicted_depth.unsqueeze(1)
            prediction = F.interpolate(
                prediction,
                size=depth.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            loss = F.smooth_l1_loss(prediction[valid], depth[valid])

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            pred = prediction.float().clamp(1e-3, 10.0)
            target = depth.float()
            count = int(valid.sum())
            ratio = torch.maximum(
                pred[valid] / target[valid], target[valid] / pred[valid]
            )
            totals["loss"] += float(loss) * count
            totals["abs_rel"] += float(
                (pred[valid] - target[valid]).abs().div(target[valid]).sum()
            )
            totals["a1"] += float((ratio < 1.25).sum())
            totals["pixels"] += count

    pixels = totals.pop("pixels")
    return {name: value / pixels for name, value in totals.items()}


def metric3d_input(rgb: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], float]:
    """공식 Metric3D v2 ViT letterbox/normalization 전처리."""
    input_height, input_width = 616, 1064
    height, width = rgb.shape[-2:]
    scale = min(input_height / height, input_width / width)
    resized_height, resized_width = int(height * scale), int(width * scale)
    rgb = F.interpolate(
        rgb * 255.0,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    mean = rgb.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]
    std = rgb.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
    rgb = (rgb - mean) / std

    pad_height = input_height - resized_height
    pad_width = input_width - resized_width
    padding = (
        pad_width // 2,
        pad_width - pad_width // 2,
        pad_height // 2,
        pad_height - pad_height // 2,
    )
    return F.pad(rgb, padding), padding, scale


def remove_padding(
    depth: torch.Tensor,
    padding: tuple[int, ...],
    output_size: tuple[int, int],
) -> torch.Tensor:
    left, right, top, bottom = padding
    depth = depth[..., top : depth.shape[-2] - bottom, left : depth.shape[-1] - right]
    return F.interpolate(
        depth.float(), size=output_size, mode="bilinear", align_corners=False
    )


def align_affine_depth(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """각 image에서 depth-space scale/shift를 최소제곱으로 정렬한다."""
    aligned = []
    for pred, gt, mask in zip(prediction, target, valid_mask):
        x, y = pred[mask], gt[mask]
        if x.numel() == 0:
            aligned.append(pred)
            continue
        scale, shift = torch.linalg.lstsq(
            torch.stack([x, torch.ones_like(x)], dim=1), y
        ).solution
        if scale < 0:
            scale = y.median() / x.median().clamp_min(1e-8)
            shift = scale.new_zeros(())
        aligned.append(pred * scale + shift)
    return torch.stack(aligned)


def depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    min_depth: float,
    max_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = prediction.clamp(min_depth, max_depth)
    target = target.clamp(min_depth, max_depth)
    counts = valid_mask.flatten(1).sum(1).clamp_min(1)
    abs_rel = torch.where(
        valid_mask,
        (prediction - target).abs() / target,
        0.0,
    ).flatten(1).sum(1) / counts
    ratio = torch.maximum(prediction / target, target / prediction)
    a1 = torch.where(valid_mask, (ratio < 1.25).float(), 0.0)
    a1 = a1.flatten(1).sum(1) / counts
    return abs_rel, a1
