import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class IQAResult:
    score: float
    gradient: float
    entropy: float
    noise: float
    mean_intensity: float

def _noise_level(channel: np.ndarray) -> float:
    image = channel.astype(np.float64, copy=False)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    flat = magnitude.reshape(-1)
    threshold = np.partition(flat, int(0.10 * (flat.size - 1)))[
        int(0.10 * (flat.size - 1))
    ]
    reliable = (
        (magnitude <= threshold)
        & (image >= 15.0)
        & (image <= 235.0)
    )
    kernel = np.array(((1, -2, 1), (-2, 4, -2), (1, -2, 1)), np.float64)
    laplacian = cv2.filter2D(image, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT)
    count = int(reliable.sum())
    height, width = image.shape
    if count < height * width * 0.0001:
        count = max((height - 2) * (width - 2), 1)
        absolute_sum = float(np.abs(laplacian).sum())
    else:
        absolute_sum = float(np.abs(laplacian[reliable]).sum())
    return math.sqrt(math.pi / 2.0) * absolute_sum / (6.0 * count)


def noise_aware_iqa(image_bgr: np.ndarray, resize_factor: float = 1.0) -> IQAResult:
    """Evaluate the paper's gradient + entropy - noise image-quality metric."""
    if image_bgr.ndim not in (2, 3) or image_bgr.size == 0:
        raise ValueError(f"Expected a non-empty grayscale/BGR image, got {image_bgr.shape}.")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit image with values in [0, 255], got {image_bgr.dtype}.")
    if not math.isfinite(resize_factor) or resize_factor <= 0:
        raise ValueError("resize_factor must be finite and positive.")

    image = image_bgr
    if resize_factor != 1.0:
        image = cv2.resize(
            image_bgr,
            None,
            fx=resize_factor,
            fy=resize_factor,
            interpolation=cv2.INTER_AREA if resize_factor < 1.0 else cv2.INTER_LINEAR,
        )
    if image.ndim == 2:
        gray = image
        channels = (image,)
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = cv2.split(image)
    else:
        raise ValueError(f"Expected one or three channels, got {image.shape}.")

    gray_float = gray.astype(np.float64, copy=False)
    grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
    normalized = cv2.magnitude(grad_x, grad_y) / math.sqrt(
        16.0 * 255.0**2 + 16.0 * 255.0**2
    )
    gamma, lambda_value = 0.06, 1_000.0
    mapped = np.zeros_like(normalized)
    selected = normalized >= gamma
    mapped[selected] = np.log(lambda_value * (normalized[selected] - gamma) + 1.0)
    mapped[selected] /= math.log(lambda_value * (1.0 - gamma) + 1.0)

    grid_size = min(10, mapped.shape[0], mapped.shape[1])
    grid_means = np.array(
        [
            float(tile.mean())
            for row in np.array_split(mapped, grid_size, axis=0)
            for tile in np.array_split(row, grid_size, axis=1)
        ],
        dtype=np.float64,
    )
    spatial_std = float(grid_means.std(ddof=1)) if grid_means.size > 1 else 0.0
    gradient = float(grid_means.mean() / spatial_std) if spatial_std > 1e-12 else 0.0

    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    entropy = 0.125 * float(-(probabilities * np.log2(probabilities)).sum())

    channel_noise = [_noise_level(channel) for channel in channels]
    if len(channel_noise) == 3:
        # OpenCV is BGR; the paper weights the green Bayer channel twice.
        noise = (channel_noise[0] + 2.0 * channel_noise[1] + channel_noise[2]) / 4.0
    else:
        noise = channel_noise[0]

    # Paper/official MATLAB hyperparameters: alpha=.4, beta=.4, Kg=2.
    score = 0.4 * 2.0 * gradient + 0.6 * entropy - 0.4 * noise
    values = (score, gradient, entropy, noise)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"The IQA metric produced a non-finite value: {values}.")
    return IQAResult(
        score=float(score),
        gradient=float(gradient),
        entropy=float(entropy),
        noise=float(noise),
        mean_intensity=float(gray_float.mean()),
    )
