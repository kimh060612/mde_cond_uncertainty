# nyu_dataset.py
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from datasets import load_dataset


class NYUv2RGBDepthDataset(Dataset):
    """
    NYU Depth v2 dataset for Depth Anything v2 Gaussian NLL fine-tuning.

    Expected source:
        Hugging Face dataset: sayakpaul/nyu_depth_v2

    Output:
        pixel_values: [3, H_in, W_in] processed by HF image_processor
        depth:        [H, W] metric depth in meters
        valid_mask:   [H, W]
    """

    def __init__(
        self,
        split: str,
        image_processor,
        image_size: Optional[Tuple[int, int]] = (518, 518),
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ):
        """
        Args:
            split:
                "train" or "validation".

            image_processor:
                AutoImageProcessor for Depth Anything v2.

            image_size:
                (height, width). If None, keep original dataset resolution.

            min_depth, max_depth:
                Valid depth range in meters.
                NYU v2 indoor depth is usually evaluated up to 10m.

            cache_dir:
                Optional Hugging Face cache directory.

            streaming:
                Keep False for normal PyTorch indexing.
                Streaming dataset does not support random access cleanly.
        """
        assert split in ["train", "validation", "val"], split

        if split == "val":
            split = "validation"

        self.dataset = load_dataset(
            "sayakpaul/nyu_depth_v2",
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

        if streaming:
            raise ValueError(
                "streaming=True is not recommended here because PyTorch Dataset "
                "requires index-based access. Use streaming=False."
            )

        self.split = split
        self.image_processor = image_processor
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __len__(self):
        return len(self.dataset)

    def _to_pil_rgb(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _to_depth_numpy(self, depth) -> np.ndarray:
        """
        HF NYU depth is usually loaded as a PIL image or ndarray.
        Depending on source preprocessing, it may already be float meters.
        This function handles common cases.

        If depth is uint16 PNG-like, assume millimeters and divide by 1000.
        If depth is float, assume meters.
        """
        if isinstance(depth, Image.Image):
            depth_np = np.array(depth)
        elif isinstance(depth, np.ndarray):
            depth_np = depth
        else:
            depth_np = np.array(depth)

        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze()

        if np.issubdtype(depth_np.dtype, np.integer):
            # NYU depth stored as integer image is commonly millimeter-like.
            depth_np = depth_np.astype(np.float32) / 1000.0
        else:
            depth_np = depth_np.astype(np.float32)

        return depth_np

    def _resize_depth_nearest(
        self,
        depth: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        h, w = size
        depth_img = Image.fromarray(depth.astype(np.float32))
        depth_img = depth_img.resize((w, h), Image.NEAREST)
        return np.array(depth_img).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]

        # sayakpaul/nyu_depth_v2 usually has "image" and "depth_map".
        # Some mirrors use "depth"; keep fallback for robustness.
        image = sample["image"]
        if "depth_map" in sample:
            depth = sample["depth_map"]
        elif "depth" in sample:
            depth = sample["depth"]
        else:
            raise KeyError(f"Cannot find depth key. Available keys: {sample.keys()}")

        image = self._to_pil_rgb(image)
        depth = self._to_depth_numpy(depth)

        if self.image_size is not None:
            h, w = self.image_size
            image = image.resize((w, h), Image.BICUBIC)
            depth = self._resize_depth_nearest(depth, self.image_size)

        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        depth = torch.from_numpy(depth).float()

        valid_mask = torch.isfinite(depth)
        valid_mask &= depth > self.min_depth
        valid_mask &= depth < self.max_depth

        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "pixel_values": pixel_values,
            "depth": depth,
            "valid_mask": valid_mask.float(),
        }


def nyu_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
    depths = torch.stack([x["depth"] for x in batch], dim=0)
    masks = torch.stack([x["valid_mask"] for x in batch], dim=0)

    return {
        "pixel_values": pixel_values,
        "depth": depths,
        "valid_mask": masks,
    }