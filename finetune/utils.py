import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    predict_depth=None,
    description="train",
    min_depth=1e-3,
    max_depth=10.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "abs_rel": 0.0, "a1": 0.0, "pixels": 0}

    progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=False)
    for batch in progress:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        if not valid.any():
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            if predict_depth is None:
                prediction = model(pixel_values=pixel_values).predicted_depth.unsqueeze(1)
                prediction = F.interpolate(
                    prediction,
                    size=depth.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)
            else:
                prediction = predict_depth(model, pixel_values, depth.shape[-2:])
            loss = F.smooth_l1_loss(prediction[valid], depth[valid])

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            pred = prediction.float().clamp(min_depth, max_depth)
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
        progress.set_postfix(
            loss=f"{totals['loss'] / totals['pixels']:.4f}",
            abs_rel=f"{totals['abs_rel'] / totals['pixels']:.4f}",
            a1=f"{totals['a1'] / totals['pixels']:.4f}",
        )

    pixels = totals.pop("pixels")
    if pixels == 0:
        raise ValueError(f"{description} contained no valid depth pixels.")
    return {name: value / pixels for name, value in totals.items()}


def run_epoch_dav3(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    predict_depth=None,
    description="train",
    min_depth=1e-3,
    max_depth=10.0,
    grad_clip=None,
    amp_enabled=None,
    amp_dtype=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "abs_rel": 0.0, "a1": 0.0, "pixels": 0}

    progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=False)
    for batch in progress:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        if not valid.any():
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)

        if amp_enabled is None:
            amp_enabled = device.type == "cuda"

        autocast_kwargs = {
            "device_type": device.type,
            "enabled": amp_enabled,
        }
        if amp_dtype is not None:
            autocast_kwargs["dtype"] = amp_dtype

        with torch.set_grad_enabled(training), torch.autocast(
            **autocast_kwargs
        ):
            focal_length_px = batch.get("focal_length_px")
            if focal_length_px is None:
                raise ValueError("DA3Metric training requires focal_length_px.")
            prediction = predict_depth(
                model,
                pixel_values,
                depth.shape[-2:],
                focal_length_px=focal_length_px.to(
                    device,
                    non_blocking=True,
                ),
            )
            prediction_fp32 = prediction.float()
            depth_fp32 = depth.float()

            loss = F.smooth_l1_loss(
                torch.log(prediction_fp32[valid].clamp_min(min_depth)),
                torch.log(depth_fp32[valid].clamp_min(min_depth)),
            )

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip if grad_clip is not None else 1.0,
            )
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            pred = prediction.float().clamp(min_depth, max_depth)
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
        progress.set_postfix(
            loss=f"{totals['loss'] / totals['pixels']:.4f}",
            abs_rel=f"{totals['abs_rel'] / totals['pixels']:.4f}",
            a1=f"{totals['a1'] / totals['pixels']:.4f}",
        )

    pixels = totals.pop("pixels")
    if pixels == 0:
        raise ValueError(f"{description} contained no valid depth pixels.")
    return {name: value / pixels for name, value in totals.items()}

def predict_metric3d(
    model,
    rgb: torch.Tensor,
    output_size: tuple[int, int],
    focal_length: float,
) -> torch.Tensor:
    model_input, padding, resize_scale = metric3d_input(rgb)
    prediction, _, _ = model({"input": model_input})
    prediction = remove_padding(prediction, padding, output_size).squeeze(1)
    return prediction * (focal_length * resize_scale / 1000.0)


def metric3d_decoder_only(model) -> torch.nn.Module:
    depth_model = getattr(model, "depth_model", None)
    if depth_model is None or not hasattr(depth_model, "encoder") or not hasattr(depth_model, "decoder"):
        raise ValueError("Metric3D model does not expose depth_model.encoder/decoder.")
    depth_model.encoder.requires_grad_(False).eval()
    depth_model.decoder.requires_grad_(True)
    return depth_model.decoder


def predict_metric3d_decoder_only(
    model,
    rgb: torch.Tensor,
    output_size: tuple[int, int],
    focal_length: float,
) -> torch.Tensor:
    model_input, padding, resize_scale = metric3d_input(rgb)
    depth_model = model.depth_model
    depth_model.encoder.eval()
    with torch.no_grad():
        features = depth_model.encoder(model_input)
    prediction = depth_model.decoder(features)["prediction"]
    prediction = remove_padding(prediction, padding, output_size).squeeze(1)
    return prediction * (focal_length * resize_scale / 1000.0)

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
