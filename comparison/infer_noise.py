from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_depth(
    model: nn.Module,
    pixel_values: torch.Tensor,
    *,
    is_dav3: bool,
    target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Run a differentiable DA2/DA3 depth forward and return BCHW depth."""
    if is_dav3:
        output = model.model(pixel_values.unsqueeze(1), export_feat_layers=[])
        depth = output["depth"]
    else:
        depth = model(pixel_values=pixel_values).predicted_depth

    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    elif depth.ndim == 5 and depth.shape[1:3] == (1, 1):
        depth = depth[:, 0]
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise RuntimeError(f"Unexpected depth shape: {tuple(depth.shape)}")
    if target_size is not None and depth.shape[-2:] != target_size:
        depth = F.interpolate(
            depth,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
    return depth


@torch.no_grad()
def infer_noise_uncertainty(
    model: nn.Module,
    pixel_values: torch.Tensor,
    *,
    is_dav3: bool,
    sample_times: int = 32,
    noise_std: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Estimate output sensitivity under repeated Gaussian input noise."""
    if sample_times < 2:
        raise ValueError("sample_times must be at least 2")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    total = square_total = None
    for _ in range(sample_times):
        noisy_pixel_values = pixel_values + torch.randn_like(pixel_values) * noise_std
        depth = predict_depth(
            model,
            noisy_pixel_values,
            is_dav3=is_dav3,
            target_size=tuple(pixel_values.shape[-2:]),
        ).float()
        total = depth if total is None else total + depth
        square = depth.square()
        square_total = square if square_total is None else square_total + square

    mean = total / sample_times
    uncertainty = (square_total / sample_times - mean.square()).clamp_min(0).sqrt()
    return {
        "mean_depth": mean,
        "uncertainty": uncertainty,
        "image_uncertainty": uncertainty.flatten(1).mean(dim=1),
    }


def _demo() -> None:
    model = nn.Sequential(nn.Conv2d(3, 1, 1, bias=False))

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return SimpleNamespace(
                predicted_depth=self.inner(pixel_values).squeeze(1)
            )

    wrapped = Wrapper(model)
    result = infer_noise_uncertainty(
        wrapped,
        torch.ones(2, 3, 4, 4),
        is_dav3=False,
        sample_times=2,
        noise_std=0.0,
    )
    assert result["image_uncertainty"].shape == (2,)
    assert torch.count_nonzero(result["uncertainty"]) == 0


if __name__ == "__main__":
    _demo()
