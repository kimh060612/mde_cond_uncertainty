from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForDepthEstimation

def _load_checkpoint_state(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(checkpoint_path), device="cpu")
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    depth_state = {
        key.removeprefix("depth_model."): value
        for key, value in state.items()
        if key.startswith("depth_model.")
    }
    if depth_state:
        state = depth_state
    return state

class CameraInducedErrorModel(nn.Module):
    """
    Frozen depth foundation model + image-level camera-induced error prediction.

    Probability model:
        scale_shift_loss(candidate_depth, canonical_depth) | x, c
            ~ N(camera_bias(x, c), var_camera(x, c))

    Outputs:
        candidate_depth
        canonical_depth
        camera_bias
        log_variance
        variance
        std
    """

    def __init__(
        self,
        model_id: str,
        context_dim: int,
        cache_dir: Optional[str] = None,
        feature_channels: int = 64,
        hidden_channels: int = 64,
        film_hidden_dim: int = 128,
        max_bias: Optional[float] = None,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
        initial_std: float = 0.5,
        variance_head_init_std: float = 1e-3,
    ) -> None:
        super().__init__()

        self.depth_model = (
            AutoModelForDepthEstimation.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
        )

        self.context_dim = context_dim
        self.max_bias = max_bias
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance
        self.initial_std = initial_std
        self.variance_head_init_std = variance_head_init_std

        # Freeze the complete foundation model.
        for parameter in self.depth_model.parameters():
            parameter.requires_grad_(False)

        self.depth_model.eval()

        config_feature_channels = getattr(
            self.depth_model.config,
            "fusion_hidden_size",
            feature_channels,
        )

        # Project frozen decoder feature into a compact shared feature.
        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                config_feature_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
        )

        # Camera context -> FiLM gamma and beta.
        self.film_generator = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(
                film_hidden_dim,
                hidden_channels * 2,
            ),
        )

        # Image-level camera-induced loss mean.
        self.bias_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        # Image-level camera-induced loss variance.
        self.variance_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        self._initialize_heads()

    @staticmethod
    def _valid_groups(
        channels: int,
        preferred_groups: int = 8,
    ) -> int:
        groups = min(channels, preferred_groups)

        while channels % groups != 0:
            groups -= 1

        return groups

    def _initialize_heads(self) -> None:
        # Initial FiLM is identity:
        # (1 + gamma) * F + beta = F
        final_film = self.film_generator[-1]
        nn.init.zeros_(final_film.weight)
        nn.init.zeros_(final_film.bias)

        # Initial camera bias is zero.
        final_bias = self.bias_head[-1]
        nn.init.zeros_(final_bias.weight)
        nn.init.zeros_(final_bias.bias)

        # Start near a chosen variance scale while retaining tiny spatial variation.
        final_variance = self.variance_head[-1]
        nn.init.normal_(
            final_variance.weight,
            mean=0.0,
            std=self.variance_head_init_std,
        )
        variance_floor = math.exp(self.min_log_variance)
        target_variance = max(self.initial_std ** 2, variance_floor + 1e-6)
        if self.max_log_variance is not None:
            target_variance = min(target_variance, math.exp(self.max_log_variance))
        softplus_target = max(target_variance - variance_floor, 1e-6)
        raw_bias = math.log(math.expm1(softplus_target))
        nn.init.constant_(final_variance.bias, raw_bias)

    def train(
        self,
        mode: bool = True,
    ) -> "CameraInducedErrorModel":
        super().train(mode)

        # Foundation model always stays in eval mode.
        self.depth_model.eval()
        return self

    def _extract_frozen_outputs(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Manually run frozen Hugging Face Depth Anything model so that both
        predicted depth and fused decoder features are available.
        """
        with torch.no_grad():
            backbone_outputs = (
                self.depth_model.backbone.forward_with_filtered_kwargs(
                    pixel_values,
                    output_hidden_states=False,
                    output_attentions=False,
                )
            )

            feature_maps = backbone_outputs.feature_maps

            _, _, height, width = pixel_values.shape

            patch_size = getattr(
                self.depth_model.config,
                "patch_size",
                14,
            )

            if isinstance(patch_size, (tuple, list)):
                patch_height = height // patch_size[0]
                patch_width = width // patch_size[1]
            else:
                patch_height = height // patch_size
                patch_width = width // patch_size

            decoder_features = self.depth_model.neck(
                feature_maps,
                patch_height,
                patch_width,
            )

            base_depth = self.depth_model.head(
                decoder_features,
                patch_height,
                patch_width,
            )

        if base_depth.ndim == 3:
            base_depth = base_depth.unsqueeze(1)

        feature_index = getattr(
            self.depth_model.config,
            "head_in_index",
            -1,
        )

        frozen_feature = decoder_features[feature_index]

        return base_depth, frozen_feature

    def _apply_film(
        self,
        feature: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(
                "context must have shape [B, context_dim]."
            )

        if context.shape[1] != self.context_dim:
            raise ValueError(
                f"Expected context_dim={self.context_dim}, "
                f"but received {context.shape[1]}."
            )

        gamma, beta = self.film_generator(context).chunk(
            2,
            dim=1,
        )

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return (1.0 + gamma) * feature + beta

    def _scalar_heads(
        self,
        conditioned_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_bias = self.bias_head(conditioned_feature).flatten(1).mean(dim=1)
        if self.max_bias is not None:
            camera_bias = self.max_bias * torch.tanh(camera_bias)

        raw_variance = self.variance_head(conditioned_feature).flatten(1).mean(dim=1)
        variance_floor = torch.exp(raw_variance.new_tensor(self.min_log_variance))
        variance = variance_floor + F.softplus(raw_variance)
        if self.max_log_variance is not None:
            variance = variance.clamp_max(torch.exp(raw_variance.new_tensor(self.max_log_variance)))

        return camera_bias, variance

    def forward(
        self,
        candidate_img: torch.Tensor,
        canonical_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        canonical_depth, _ = self._extract_frozen_outputs(canonical_img) 

        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)

        camera_bias, variance = self._scalar_heads(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)
        canonical_depth = F.interpolate(canonical_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "canonical_depth": canonical_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }

    def inference(
        self,
        candidate_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ):
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)
        
        camera_bias, variance = self._scalar_heads(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }

class CameraInducedErrorModelDAv3(nn.Module):
    """
    Frozen depth foundation model + image-level camera-induced error prediction.

    Probability model:
        scale_shift_loss(candidate_depth, canonical_depth) | x, c
            ~ N(camera_bias(x, c), var_camera(x, c))

    Outputs:
        candidate_depth
        canonical_depth
        camera_bias
        log_variance
        variance
        std
    """

    def __init__(
        self,
        model_id: str,
        context_dim: int,
        cache_dir: Optional[str] = None,
        feature_channels: int = 64,
        hidden_channels: int = 64,
        film_hidden_dim: int = 128,
        max_bias: Optional[float] = None,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
        initial_std: float = 0.5,
        variance_head_init_std: float = 1e-3,
        canonical_group_size: int = 1,
    ) -> None:
        super().__init__()

        if canonical_group_size < 1:
            raise ValueError("canonical_group_size must be at least 1.")

        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "CameraInducedErrorModelDAv3 requires the depth-anything-3 package."
            ) from exc

        load_kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
        self.depth_model = DepthAnything3.from_pretrained(model_id, **load_kwargs)

        self.context_dim = context_dim
        self.max_bias = max_bias
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance
        self.initial_std = initial_std
        self.variance_head_init_std = variance_head_init_std
        self.canonical_group_size = int(canonical_group_size)

        # Freeze the complete foundation model.
        for parameter in self.depth_model.parameters():
            parameter.requires_grad_(False)

        self.depth_model.eval()

        decoder_output = self.depth_model.model.head.scratch.output_conv1
        config_feature_channels = getattr(decoder_output, "out_channels", feature_channels)

        # Project frozen decoder feature into a compact shared feature.
        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                config_feature_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
        )

        # Camera context -> FiLM gamma and beta.
        self.film_generator = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(
                film_hidden_dim,
                hidden_channels * 2,
            ),
        )

        # Image-level camera-induced loss mean.
        self.bias_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        # Image-level camera-induced loss variance.
        self.variance_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        self._initialize_heads()

    @staticmethod
    def _valid_groups(
        channels: int,
        preferred_groups: int = 8,
    ) -> int:
        groups = min(channels, preferred_groups)

        while channels % groups != 0:
            groups -= 1

        return groups

    def _initialize_heads(self) -> None:
        # Initial FiLM is identity:
        # (1 + gamma) * F + beta = F
        final_film = self.film_generator[-1]
        nn.init.zeros_(final_film.weight)
        nn.init.zeros_(final_film.bias)

        # Initial camera bias is zero.
        final_bias = self.bias_head[-1]
        nn.init.zeros_(final_bias.weight)
        nn.init.zeros_(final_bias.bias)

        # Start near a chosen variance scale while retaining tiny spatial variation.
        final_variance = self.variance_head[-1]
        nn.init.normal_(
            final_variance.weight,
            mean=0.0,
            std=self.variance_head_init_std,
        )
        variance_floor = math.exp(self.min_log_variance)
        target_variance = max(self.initial_std ** 2, variance_floor + 1e-6)
        if self.max_log_variance is not None:
            target_variance = min(target_variance, math.exp(self.max_log_variance))
        softplus_target = max(target_variance - variance_floor, 1e-6)
        raw_bias = math.log(math.expm1(softplus_target))
        nn.init.constant_(final_variance.bias, raw_bias)

    def train(
        self,
        mode: bool = True,
    ) -> "CameraInducedErrorModelDAv3":
        super().train(mode)

        # Foundation model always stays in eval mode.
        self.depth_model.eval()
        return self

    def _extract_frozen_outputs(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return DA3 single-view depth and its final fused decoder feature."""
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape [B, 3, H, W].")

        fused_feature = None

        def capture_fused_feature(_module, _inputs, output):
            nonlocal fused_feature
            fused_feature = output

        decoder_output = self.depth_model.model.head.scratch.output_conv1
        hook = decoder_output.register_forward_hook(capture_fused_feature)
        try:
            output = self.depth_model(pixel_values.unsqueeze(1), export_feat_layers=[])
        finally:
            hook.remove()

        if fused_feature is None:
            raise RuntimeError("Depth Anything 3 decoder feature was not produced.")

        base_depth = output["depth"]
        if base_depth.ndim == 3:
            base_depth = base_depth.unsqueeze(1)
        elif base_depth.ndim == 5 and base_depth.shape[1:3] == (1, 1):
            base_depth = base_depth[:, 0]

        if fused_feature.ndim == 5 and fused_feature.shape[1] == 1:
            fused_feature = fused_feature[:, 0]

        if base_depth.ndim != 4 or base_depth.shape[1] != 1:
            raise RuntimeError(f"Unexpected DA3 depth shape: {tuple(base_depth.shape)}")
        if fused_feature.ndim != 4:
            raise RuntimeError(f"Unexpected DA3 decoder feature shape: {tuple(fused_feature.shape)}")

        # DA3.forward uses inference_mode; clone to make tensors safe for trainable heads.
        return base_depth.float().clone(), fused_feature.float().clone()

    def _apply_film(
        self,
        feature: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(
                "context must have shape [B, context_dim]."
            )

        if context.shape[1] != self.context_dim:
            raise ValueError(
                f"Expected context_dim={self.context_dim}, "
                f"but received {context.shape[1]}."
            )

        gamma, beta = self.film_generator(context).chunk(
            2,
            dim=1,
        )

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return (1.0 + gamma) * feature + beta

    def _scalar_heads(
        self,
        conditioned_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_bias = self.bias_head(conditioned_feature).flatten(1).mean(dim=1)
        if self.max_bias is not None:
            camera_bias = self.max_bias * torch.tanh(camera_bias)

        raw_variance = self.variance_head(conditioned_feature).flatten(1).mean(dim=1)
        variance_floor = torch.exp(raw_variance.new_tensor(self.min_log_variance))
        variance = variance_floor + F.softplus(raw_variance)
        if self.max_log_variance is not None:
            variance = variance.clamp_max(torch.exp(raw_variance.new_tensor(self.max_log_variance)))

        return camera_bias, variance

    def forward(
        self,
        candidate_img: torch.Tensor,
        canonical_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        if canonical_img.shape[0] != candidate_img.shape[0]:
            raise ValueError("candidate_img and canonical_img must have the same batch size.")
        if canonical_img.shape[0] % self.canonical_group_size != 0:
            raise ValueError(
                "canonical batch size must be divisible by canonical_group_size."
            )

        canonical_depth, _ = self._extract_frozen_outputs(
            canonical_img[::self.canonical_group_size]
        )
        canonical_depth = canonical_depth.repeat_interleave(
            self.canonical_group_size,
            dim=0,
        )

        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)

        camera_bias, variance = self._scalar_heads(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)
        canonical_depth = F.interpolate(canonical_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "canonical_depth": canonical_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }

    def inference(
        self,
        candidate_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ):
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)
        
        camera_bias, variance = self._scalar_heads(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }

class CameraInducedErrorMetricModel(nn.Module):
    """
    Frozen depth foundation model + image-level camera-induced error prediction.

    Probability model:
        scale_shift_loss(candidate_depth, canonical_depth) | x, c
            ~ N(camera_bias(x, c), var_camera(x, c))

    Outputs:
        candidate_depth
        canonical_depth
        camera_bias
        log_variance
        variance
        std
    """

    def __init__(
        self,
        model_id: str,
        context_dim: int,
        checkpoint_path: Path,
        cache_dir: Optional[str] = None,
        feature_channels: int = 64,
        hidden_channels: int = 64,
        film_hidden_dim: int = 128,
        max_bias: Optional[float] = None,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
        initial_std: float = 0.5,
        variance_head_init_std: float = 1e-3,
    ) -> None:
        super().__init__()

        self.depth_model = (
            AutoModelForDepthEstimation.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
        )
        local_model_dir = checkpoint_path if checkpoint_path.is_dir() else None
        if checkpoint_path.is_file() and not (
            local_model_dir is not None
            and checkpoint_path.name in {"model.safetensors", "pytorch_model.bin"}
        ):
            self.depth_model.load_state_dict(_load_checkpoint_state(checkpoint_path), strict=True)

        self.context_dim = context_dim
        self.max_bias = max_bias
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance
        self.initial_std = initial_std
        self.variance_head_init_std = variance_head_init_std

        # Freeze the complete foundation model.
        for parameter in self.depth_model.parameters():
            parameter.requires_grad_(False)

        self.depth_model.eval()

        config_feature_channels = getattr(
            self.depth_model.config,
            "fusion_hidden_size",
            feature_channels,
        )

        # Project frozen decoder feature into a compact shared feature.
        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                config_feature_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
        )

        # Camera context -> FiLM gamma and beta.
        self.film_generator = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(
                film_hidden_dim,
                hidden_channels * 2,
            ),
        )

        # Image-level camera-induced loss mean.
        self.bias_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        # Image-level camera-induced loss variance.
        self.variance_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        self._initialize_heads()

    @staticmethod
    def _valid_groups(
        channels: int,
        preferred_groups: int = 8,
    ) -> int:
        groups = min(channels, preferred_groups)

        while channels % groups != 0:
            groups -= 1

        return groups

    def _initialize_heads(self) -> None:
        # Initial FiLM is identity:
        # (1 + gamma) * F + beta = F
        final_film = self.film_generator[-1]
        nn.init.zeros_(final_film.weight)
        nn.init.zeros_(final_film.bias)

        # Initial camera bias is zero.
        final_bias = self.bias_head[-1]
        nn.init.zeros_(final_bias.weight)
        nn.init.zeros_(final_bias.bias)

        # Start near a chosen variance scale while retaining tiny spatial variation.
        final_variance = self.variance_head[-1]
        nn.init.normal_(
            final_variance.weight,
            mean=0.0,
            std=self.variance_head_init_std,
        )
        variance_floor = math.exp(self.min_log_variance)
        target_variance = max(self.initial_std ** 2, variance_floor + 1e-6)
        if self.max_log_variance is not None:
            target_variance = min(target_variance, math.exp(self.max_log_variance))
        softplus_target = max(target_variance - variance_floor, 1e-6)
        raw_bias = math.log(math.expm1(softplus_target))
        nn.init.constant_(final_variance.bias, raw_bias)

    def train(
        self,
        mode: bool = True,
    ) -> "CameraInducedErrorModel":
        super().train(mode)

        # Foundation model always stays in eval mode.
        self.depth_model.eval()
        return self

    def _extract_frozen_outputs(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Manually run frozen Hugging Face Depth Anything model so that both
        predicted depth and fused decoder features are available.
        """
        with torch.no_grad():
            backbone_outputs = (
                self.depth_model.backbone.forward_with_filtered_kwargs(
                    pixel_values,
                    output_hidden_states=False,
                    output_attentions=False,
                )
            )

            feature_maps = backbone_outputs.feature_maps

            _, _, height, width = pixel_values.shape

            patch_size = getattr(
                self.depth_model.config,
                "patch_size",
                14,
            )

            if isinstance(patch_size, (tuple, list)):
                patch_height = height // patch_size[0]
                patch_width = width // patch_size[1]
            else:
                patch_height = height // patch_size
                patch_width = width // patch_size

            decoder_features = self.depth_model.neck(
                feature_maps,
                patch_height,
                patch_width,
            )

            base_depth = self.depth_model.head(
                decoder_features,
                patch_height,
                patch_width,
            )

        if base_depth.ndim == 3:
            base_depth = base_depth.unsqueeze(1)

        feature_index = getattr(
            self.depth_model.config,
            "head_in_index",
            -1,
        )

        frozen_feature = decoder_features[feature_index]

        return base_depth, frozen_feature

    def _apply_film(
        self,
        feature: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(
                "context must have shape [B, context_dim]."
            )

        if context.shape[1] != self.context_dim:
            raise ValueError(
                f"Expected context_dim={self.context_dim}, "
                f"but received {context.shape[1]}."
            )

        gamma, beta = self.film_generator(context).chunk(
            2,
            dim=1,
        )

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return (1.0 + gamma) * feature + beta

    def _scalar_heads(
        self,
        conditioned_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_bias = self.bias_head(conditioned_feature).flatten(1).mean(dim=1)
        if self.max_bias is not None:
            camera_bias = self.max_bias * torch.tanh(camera_bias)

        raw_variance = self.variance_head(conditioned_feature).flatten(1).mean(dim=1)
        variance_floor = torch.exp(raw_variance.new_tensor(self.min_log_variance))
        variance = variance_floor + F.softplus(raw_variance)
        if self.max_log_variance is not None:
            variance = variance.clamp_max(torch.exp(raw_variance.new_tensor(self.max_log_variance)))

        return camera_bias, variance

    def forward(
        self,
        candidate_img: torch.Tensor,
        canonical_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        canonical_depth, _ = self._extract_frozen_outputs(canonical_img) 

        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)

        camera_bias, variance = self._scalar_heads(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)
        canonical_depth = F.interpolate(canonical_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "canonical_depth": canonical_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }

    def inference(
        self,
        candidate_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ):
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)
        
        camera_bias, variance = self._scalar_heads(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }

class CameraInducedErrorMetricModelDAv3(nn.Module):
    """
    Frozen depth foundation model + image-level camera-induced error prediction.

    Probability model:
        scale_shift_loss(candidate_depth, canonical_depth) | x, c
            ~ N(camera_bias(x, c), var_camera(x, c))

    Outputs:
        candidate_depth
        canonical_depth
        camera_bias
        log_variance
        variance
        std
    """

    def __init__(
        self,
        model_id: str,
        context_dim: int,
        checkpoint_path: Path, 
        cache_dir: Optional[str] = None,
        feature_channels: int = 64,
        hidden_channels: int = 64,
        film_hidden_dim: int = 128,
        max_bias: Optional[float] = None,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
        initial_std: float = 0.5,
        variance_head_init_std: float = 1e-3,
        canonical_group_size: int = 1,
    ) -> None:
        super().__init__()

        if canonical_group_size < 1:
            raise ValueError("canonical_group_size must be at least 1.")

        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "CameraInducedErrorModelDAv3 requires the depth-anything-3 package."
            ) from exc

        load_kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
        self.depth_model = DepthAnything3.from_pretrained(model_id, **load_kwargs)
        local_model_dir = checkpoint_path if checkpoint_path.is_dir() else None
        if checkpoint_path.is_file() and not (
            local_model_dir is not None
            and checkpoint_path.name in {"model.safetensors", "pytorch_model.bin"}
        ):
            state = _load_checkpoint_state(checkpoint_path)
            target = self.depth_model if any(key.startswith("model.") for key in state) else self.depth_model.model
            target.load_state_dict(state, strict=True)

        self.context_dim = context_dim
        self.max_bias = max_bias
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance
        self.initial_std = initial_std
        self.variance_head_init_std = variance_head_init_std
        self.canonical_group_size = int(canonical_group_size)

        # Freeze the complete foundation model.
        for parameter in self.depth_model.parameters():
            parameter.requires_grad_(False)

        self.depth_model.eval()

        decoder_output = self.depth_model.model.head.scratch.output_conv1
        config_feature_channels = getattr(decoder_output, "out_channels", feature_channels)

        # Project frozen decoder feature into a compact shared feature.
        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                config_feature_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
        )

        # Camera context -> FiLM gamma and beta.
        self.film_generator = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(
                film_hidden_dim,
                hidden_channels * 2,
            ),
        )

        # Image-level camera-induced loss mean.
        self.bias_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        # Image-level camera-induced loss variance.
        self.variance_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        self._initialize_heads()

    @staticmethod
    def _valid_groups(
        channels: int,
        preferred_groups: int = 8,
    ) -> int:
        groups = min(channels, preferred_groups)

        while channels % groups != 0:
            groups -= 1

        return groups

    def _initialize_heads(self) -> None:
        # Initial FiLM is identity:
        # (1 + gamma) * F + beta = F
        final_film = self.film_generator[-1]
        nn.init.zeros_(final_film.weight)
        nn.init.zeros_(final_film.bias)

        # Initial camera bias is zero.
        final_bias = self.bias_head[-1]
        nn.init.zeros_(final_bias.weight)
        nn.init.zeros_(final_bias.bias)

        # Start near a chosen variance scale while retaining tiny spatial variation.
        final_variance = self.variance_head[-1]
        nn.init.normal_(
            final_variance.weight,
            mean=0.0,
            std=self.variance_head_init_std,
        )
        variance_floor = math.exp(self.min_log_variance)
        target_variance = max(self.initial_std ** 2, variance_floor + 1e-6)
        if self.max_log_variance is not None:
            target_variance = min(target_variance, math.exp(self.max_log_variance))
        softplus_target = max(target_variance - variance_floor, 1e-6)
        raw_bias = math.log(math.expm1(softplus_target))
        nn.init.constant_(final_variance.bias, raw_bias)

    def train(
        self,
        mode: bool = True,
    ) -> "CameraInducedErrorModelDAv3":
        super().train(mode)

        # Foundation model always stays in eval mode.
        self.depth_model.eval()
        return self

    def _extract_frozen_outputs(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return DA3 single-view depth and its final fused decoder feature."""
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape [B, 3, H, W].")

        fused_feature = None

        def capture_fused_feature(_module, _inputs, output):
            nonlocal fused_feature
            fused_feature = output

        decoder_output = self.depth_model.model.head.scratch.output_conv1
        hook = decoder_output.register_forward_hook(capture_fused_feature)
        try:
            output = self.depth_model(pixel_values.unsqueeze(1), export_feat_layers=[])
        finally:
            hook.remove()

        if fused_feature is None:
            raise RuntimeError("Depth Anything 3 decoder feature was not produced.")

        base_depth = output["depth"]
        if base_depth.ndim == 3:
            base_depth = base_depth.unsqueeze(1)
        elif base_depth.ndim == 5 and base_depth.shape[1:3] == (1, 1):
            base_depth = base_depth[:, 0]

        if fused_feature.ndim == 5 and fused_feature.shape[1] == 1:
            fused_feature = fused_feature[:, 0]

        if base_depth.ndim != 4 or base_depth.shape[1] != 1:
            raise RuntimeError(f"Unexpected DA3 depth shape: {tuple(base_depth.shape)}")
        if fused_feature.ndim != 4:
            raise RuntimeError(f"Unexpected DA3 decoder feature shape: {tuple(fused_feature.shape)}")

        # DA3.forward uses inference_mode; clone to make tensors safe for trainable heads.
        return base_depth.float().clone(), fused_feature.float().clone()

    def _apply_film(
        self,
        feature: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(
                "context must have shape [B, context_dim]."
            )

        if context.shape[1] != self.context_dim:
            raise ValueError(
                f"Expected context_dim={self.context_dim}, "
                f"but received {context.shape[1]}."
            )

        gamma, beta = self.film_generator(context).chunk(
            2,
            dim=1,
        )

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return (1.0 + gamma) * feature + beta

    def _scalar_heads(
        self,
        conditioned_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_bias = self.bias_head(conditioned_feature).flatten(1).mean(dim=1)
        if self.max_bias is not None:
            camera_bias = self.max_bias * torch.tanh(camera_bias)

        raw_variance = self.variance_head(conditioned_feature).flatten(1).mean(dim=1)
        variance_floor = torch.exp(raw_variance.new_tensor(self.min_log_variance))
        variance = variance_floor + F.softplus(raw_variance)
        if self.max_log_variance is not None:
            variance = variance.clamp_max(torch.exp(raw_variance.new_tensor(self.max_log_variance)))

        return camera_bias, variance

    def forward(
        self,
        candidate_img: torch.Tensor,
        canonical_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        if canonical_img.shape[0] != candidate_img.shape[0]:
            raise ValueError("candidate_img and canonical_img must have the same batch size.")
        if canonical_img.shape[0] % self.canonical_group_size != 0:
            raise ValueError(
                "canonical batch size must be divisible by canonical_group_size."
            )

        canonical_depth, _ = self._extract_frozen_outputs(
            canonical_img[::self.canonical_group_size]
        )
        canonical_depth = canonical_depth.repeat_interleave(
            self.canonical_group_size,
            dim=0,
        )

        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)

        camera_bias, variance = self._scalar_heads(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)
        canonical_depth = F.interpolate(canonical_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "canonical_depth": canonical_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }

    def inference(
        self,
        candidate_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ):
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)
        
        camera_bias, variance = self._scalar_heads(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }




# class CameraInducedErrorModelRGBInput(nn.Module):
#     def __init__(
#         self,
#         context_dim: int,
#         feature_channels: int = 128,
#         hidden_channels: int = 64,
#         film_hidden_dim: int = 128,
#         max_bias: Optional[float] = None,
#         min_log_variance: float = -10.0,
#         max_log_variance: float = 10.0,
#         initial_std: float = 0.5,
#         variance_head_init_std: float = 1e-3,
#     ):
#         super().__init__()

#         self.context_dim = context_dim
#         self.max_bias = max_bias
#         self.min_log_variance = min_log_variance
#         self.initial_std = initial_std
#         self.max_log_variance = max_log_variance
#         self.variance_head_init_std = variance_head_init_std

#         self.image_encoder = nn.Sequential(
#             nn.Conv2d(3, feature_channels // 4, 3, stride=2, padding=1, bias=False),
#             nn.GroupNorm(self._valid_groups(feature_channels // 4),feature_channels // 4),
#             nn.GELU(),
#             nn.Conv2d(feature_channels // 4,feature_channels // 4,3,stride=2,padding=1,bias=False),
#             nn.GroupNorm(self._valid_groups(feature_channels // 2),feature_channels // 4),
#             nn.GELU(),
#             nn.Conv2d(feature_channels // 4,feature_channels // 2,3,stride=2,padding=1,bias=False),
#             nn.GroupNorm(self._valid_groups(feature_channels // 2),feature_channels // 2),
#             nn.GELU(),
#             nn.Conv2d(feature_channels // 2,feature_channels,3,stride=2,padding=1,bias=False),
#             nn.GroupNorm(self._valid_groups(feature_channels),feature_channels),
#             nn.GELU(),
#         )

#         # Project encoded RGB feature into a compact shared feature.
#         self.feature_projection = nn.Sequential(
#             nn.Conv2d(feature_channels,hidden_channels,kernel_size=3,padding=1,bias=False),
#             nn.GroupNorm(self._valid_groups(hidden_channels),hidden_channels),
#             nn.GELU(),
#         )

#         # Camera context -> FiLM gamma and beta.
#         self.film_generator = nn.Sequential(
#             nn.LayerNorm(context_dim),
#             nn.Linear(context_dim, film_hidden_dim),
#             nn.GELU(),
#             nn.Linear(film_hidden_dim,hidden_channels * 2),
#         )

#         # Image-level camera-induced loss mean.
#         self.bias_head = nn.Sequential(
#             nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False),
#             nn.GroupNorm(self._valid_groups(hidden_channels),hidden_channels),
#             nn.GELU(),
#             nn.Conv2d(hidden_channels,hidden_channels // 2,kernel_size=3,padding=1),
#             nn.GELU(),
#             nn.Conv2d(hidden_channels // 2,1,kernel_size=1),
#         )

#         # Image-level camera-induced loss variance.
#         self.variance_head = nn.Sequential(
#             nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,padding=1,bias=False),
#             nn.GroupNorm(self._valid_groups(hidden_channels),hidden_channels),
#             nn.GELU(),
#             nn.Conv2d(hidden_channels,hidden_channels // 2,kernel_size=3,padding=1),
#             nn.GELU(),
#             nn.Conv2d(hidden_channels // 2,1,kernel_size=1),
#         )
#         self._initialize_heads()

#     @staticmethod
#     def _valid_groups(
#         channels: int,
#         preferred_groups: int = 8,
#     ) -> int:
#         groups = min(channels, preferred_groups)

#         while channels % groups != 0:
#             groups -= 1

#         return groups

#     def _initialize_heads(self) -> None:
#         # Initial FiLM is identity:
#         # (1 + gamma) * F + beta = F
#         final_film = self.film_generator[-1]
#         nn.init.zeros_(final_film.weight)
#         nn.init.zeros_(final_film.bias)

#         # Initial camera bias is zero.
#         final_bias = self.bias_head[-1]
#         nn.init.zeros_(final_bias.weight)
#         nn.init.zeros_(final_bias.bias)

#         # Start near a chosen variance scale while retaining tiny spatial variation.
#         final_variance = self.variance_head[-1]
#         nn.init.normal_(
#             final_variance.weight,
#             mean=0.0,
#             std=self.variance_head_init_std,
#         )
#         variance_floor = math.exp(self.min_log_variance)
#         target_variance = max(self.initial_std ** 2, variance_floor + 1e-6)
#         if self.max_log_variance is not None:
#             target_variance = min(target_variance, math.exp(self.max_log_variance))
#         softplus_target = max(target_variance - variance_floor, 1e-6)
#         raw_bias = math.log(math.expm1(softplus_target))
#         nn.init.constant_(final_variance.bias, raw_bias)
    
#     def _predict(
#         self,
#         image: torch.Tensor,
#         context: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         feature = self.feature_projection(self.image_encoder(image))
#         gamma, beta = self.film_generator(context).chunk(2, dim=1)
#         feature = (1.0 + gamma[:, :, None, None]) * feature + beta[:, :, None, None]

#         camera_bias = self.bias_head(feature).flatten(1).mean(dim=1)
#         if self.max_bias is not None:
#             camera_bias = self.max_bias * torch.tanh(camera_bias)

#         raw_variance = self.variance_head(feature).flatten(1).mean(dim=1)
#         variance = torch.exp(
#             raw_variance.new_tensor(self.min_log_variance)
#         ) + F.softplus(raw_variance)
#         if self.max_log_variance is not None:
#             variance = variance.clamp_max(
#                 torch.exp(raw_variance.new_tensor(self.max_log_variance))
#             )

#         return {
#             "predicted_loss": camera_bias,
#             "camera_bias": camera_bias,
#             "log_variance": torch.log(variance.clamp_min(1e-8)),
#             "variance": variance,
#             "std": torch.sqrt(variance),
#         }

#     def forward(
#         self,
#         candidate_img: torch.Tensor,
#         canonical_img: torch.Tensor,
#         context: torch.Tensor,
#         target_size: Optional[tuple[int, int]] = None,
#     ) -> Dict[str, torch.Tensor]:
#         return self._predict(candidate_img, context)

#     def inference(
#         self,
#         candidate_img: torch.Tensor,
#         context: torch.Tensor,
#         target_size: Optional[tuple[int, int]] = None,
#     ) -> Dict[str, torch.Tensor]:
#         return self._predict(candidate_img, context)

# def forward_with_rgb_model(
#     model: CameraInducedErrorModelRGBInput,
#     mde_model: nn.Module,
#     candidate_img: torch.Tensor,
#     canonical_img: torch.Tensor,
#     context: torch.Tensor,
#     target_size: Optional[tuple[int, int]] = None,
# ) -> Dict[str, torch.Tensor]:
#     with torch.no_grad():
#         candidate_depth = mde_model(pixel_values=candidate_img).predicted_depth
#         canonical_depth = mde_model(pixel_values=canonical_img).predicted_depth
#         if target_size is None:
#             target_size = candidate_depth.shape[-2:]
#         candidate_depth = F.interpolate(candidate_depth.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False)
#         canonical_depth = F.interpolate(canonical_depth.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False)
#     predictions = model.forward(candidate_img, canonical_img, context, target_size)
#     return {
#         "candidate_depth": candidate_depth,
#         "canonical_depth": canonical_depth,
#         **predictions,
#     }

# def inference_with_rgb_model(
#     model: CameraInducedErrorModelRGBInput,
#     mde_model: nn.Module,
#     candidate_img: torch.Tensor,
#     context: torch.Tensor,
#     target_size: Optional[tuple[int, int]] = None,
# ) -> Dict[str, torch.Tensor]:
#     with torch.no_grad():
#         candidate_depth = mde_model(pixel_values=candidate_img).predicted_depth
#         if target_size is None:
#             target_size = candidate_depth.shape[-2:]
#         candidate_depth = F.interpolate(candidate_depth.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False)
#     predictions = model.inference(candidate_img, context, target_size)
#     return {
#         "candidate_depth": candidate_depth,
#         **predictions,
#     }

class CameraInducedErrorDecompositionModel(nn.Module):
    """
    Frozen depth foundation model + image-level camera-induced error prediction.

    Probability model:
        scale_shift_loss(candidate_depth, canonical_depth) | x, c
            ~ N(camera_bias(x, c), var_camera(x, c))

    Outputs:
        candidate_depth
        canonical_depth
        camera_bias
        log_variance
        variance
        std
    """

    def __init__(
        self,
        model_id: str,
        context_dim: int,
        cache_dir: Optional[str] = None,
        feature_channels: int = 64,
        hidden_channels: int = 64,
        film_hidden_dim: int = 128,
        max_bias: Optional[float] = None,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
        initial_std: float = 0.5,
        variance_head_init_std: float = 1e-3,
    ) -> None:
        super().__init__()

        self.depth_model = (
            AutoModelForDepthEstimation.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
        )

        self.context_dim = context_dim
        self.max_bias = max_bias
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance
        self.initial_std = initial_std
        self.variance_head_init_std = variance_head_init_std

        # Freeze the complete foundation model.
        for parameter in self.depth_model.parameters():
            parameter.requires_grad_(False)

        self.depth_model.eval()

        config_feature_channels = getattr(
            self.depth_model.config,
            "fusion_hidden_size",
            feature_channels,
        )

        # Project frozen decoder feature into a compact shared feature.
        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                config_feature_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
        )

        # Camera context -> FiLM gamma and beta.
        self.film_generator = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, film_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(film_hidden_dim),
            nn.Linear(film_hidden_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(
                film_hidden_dim,
                hidden_channels * 2,
            ),
        )

        # Image-level camera-induced loss mean.
        self.camera_bias_head = nn.Sequential(
            nn.Linear(context_dim, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        
        self.scene_camera_bias_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        # Image-level camera-induced loss variance.
        self.variance_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._valid_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

        self._initialize_heads()

    @staticmethod
    def _valid_groups(
        channels: int,
        preferred_groups: int = 8,
    ) -> int:
        groups = min(channels, preferred_groups)

        while channels % groups != 0:
            groups -= 1

        return groups

    def _initialize_heads(self) -> None:
        # Initial FiLM is identity:
        # (1 + gamma) * F + beta = F
        final_film = self.film_generator[-1]
        nn.init.zeros_(final_film.weight)
        nn.init.zeros_(final_film.bias)

        # Initial camera bias is zero.
        final_bias = self.scene_camera_bias_head[-1]
        nn.init.zeros_(final_bias.weight)
        nn.init.zeros_(final_bias.bias)
        final_camera_bias = self.camera_bias_head[-1]
        nn.init.zeros_(final_camera_bias.weight)
        nn.init.zeros_(final_camera_bias.bias)

        # Start near a chosen variance scale while retaining tiny spatial variation.
        final_variance = self.variance_head[-1]
        nn.init.normal_(
            final_variance.weight,
            mean=0.0,
            std=self.variance_head_init_std,
        )
        variance_floor = math.exp(self.min_log_variance)
        target_variance = max(self.initial_std ** 2, variance_floor + 1e-6)
        if self.max_log_variance is not None:
            target_variance = min(target_variance, math.exp(self.max_log_variance))
        softplus_target = max(target_variance - variance_floor, 1e-6)
        raw_bias = math.log(math.expm1(softplus_target))
        nn.init.constant_(final_variance.bias, raw_bias)

    def train(
        self,
        mode: bool = True,
    ) -> "CameraInducedErrorModel":
        super().train(mode)

        # Foundation model always stays in eval mode.
        self.depth_model.eval()
        return self

    def _extract_frozen_outputs(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Manually run frozen Hugging Face Depth Anything model so that both
        predicted depth and fused decoder features are available.
        """
        with torch.no_grad():
            backbone_outputs = (
                self.depth_model.backbone.forward_with_filtered_kwargs(
                    pixel_values,
                    output_hidden_states=False,
                    output_attentions=False,
                )
            )

            feature_maps = backbone_outputs.feature_maps

            _, _, height, width = pixel_values.shape

            patch_size = getattr(
                self.depth_model.config,
                "patch_size",
                14,
            )

            if isinstance(patch_size, (tuple, list)):
                patch_height = height // patch_size[0]
                patch_width = width // patch_size[1]
            else:
                patch_height = height // patch_size
                patch_width = width // patch_size

            decoder_features = self.depth_model.neck(
                feature_maps,
                patch_height,
                patch_width,
            )

            base_depth = self.depth_model.head(
                decoder_features,
                patch_height,
                patch_width,
            )

        if base_depth.ndim == 3:
            base_depth = base_depth.unsqueeze(1)

        feature_index = getattr(
            self.depth_model.config,
            "head_in_index",
            -1,
        )

        frozen_feature = decoder_features[feature_index]

        return base_depth, frozen_feature

    def _apply_film(
        self,
        feature: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(
                "context must have shape [B, context_dim]."
            )

        if context.shape[1] != self.context_dim:
            raise ValueError(
                f"Expected context_dim={self.context_dim}, "
                f"but received {context.shape[1]}."
            )

        gamma, beta = self.film_generator(context).chunk(
            2,
            dim=1,
        )

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return (1.0 + gamma) * feature + beta
    
    def _scalar_heads(
        self,
        conditioned_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_bias = self.scene_camera_bias_head(conditioned_feature).flatten(1).mean(dim=1)
        raw_variance = self.variance_head(conditioned_feature).flatten(1).mean(dim=1)
        variance_floor = torch.exp(raw_variance.new_tensor(self.min_log_variance))
        variance = variance_floor + F.softplus(raw_variance)
        if self.max_log_variance is not None:
            variance = variance.clamp_max(torch.exp(raw_variance.new_tensor(self.max_log_variance)))

        return camera_bias, variance

    def forward(
        self,
        candidate_img: torch.Tensor,
        canonical_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        canonical_depth, _ = self._extract_frozen_outputs(canonical_img) 

        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)

        scene_camera_bias, variance = self._scalar_heads(conditioned_feature)
        camera_only_bias = self.camera_bias_head(context).squeeze(-1)
        camera_bias = scene_camera_bias + camera_only_bias # \Delta = \Delta_{camera} + \Delta_{scene x bias}
        if self.max_bias is not None:
            camera_bias = self.max_bias * torch.tanh(camera_bias)
        
        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)
        canonical_depth = F.interpolate(canonical_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "canonical_depth": canonical_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }

    def inference(
        self,
        candidate_img: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ):
        candidate_depth, frozen_feature = self._extract_frozen_outputs(candidate_img)
        shared_feature = self.feature_projection(
            frozen_feature
        )
        conditioned_feature = self._apply_film(shared_feature, context)
        
        scene_camera_bias, variance = self._scalar_heads(conditioned_feature)
        camera_only_bias = self.camera_bias_head(context).squeeze(-1)
        camera_bias = scene_camera_bias + camera_only_bias # \Delta = \Delta_{camera} + \Delta_{scene x bias}
        if self.max_bias is not None:
            camera_bias = self.max_bias * torch.tanh(camera_bias)
        
        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }
