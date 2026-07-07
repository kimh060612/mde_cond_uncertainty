from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation


class CameraInducedErrorModel(nn.Module):
    """
    Frozen depth foundation model + camera-induced error prediction & uncertainty.

    Probability model:
        y | x, c ~ N(
            mu_base(x) + bias_camera(x, c),
            var_camera(x, c)
        )

    Outputs:
        base_depth
        camera_bias
        corrected_depth
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

        # Camera-induced mean correction.
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

        # Camera-induced log variance.
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

        camera_bias = self.bias_head(conditioned_feature)
        if self.max_bias is not None:
            camera_bias = self.max_bias * torch.tanh(camera_bias)
        raw_variance = self.variance_head(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)
        canonical_depth = F.interpolate(canonical_depth, size=target_size, mode="bilinear", align_corners=False)
        camera_bias = F.interpolate(camera_bias, size=target_size, mode="bilinear", align_corners=False)
        raw_variance = F.interpolate(raw_variance, size=target_size, mode="bilinear", align_corners=False)

        variance_floor = torch.exp(raw_variance.new_tensor(self.min_log_variance))
        variance = variance_floor + F.softplus(raw_variance)
        if self.max_log_variance is not None:
            variance = variance.clamp_max(torch.exp(raw_variance.new_tensor(self.max_log_variance)))

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "canonical_depth": canonical_depth,
            "corrected_depth": candidate_depth + camera_bias,
            "camera_bias": camera_bias,
            "raw_variance": raw_variance,
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
        
        camera_bias = self.bias_head(conditioned_feature)
        if self.max_bias is not None:
            camera_bias = self.max_bias * torch.tanh(camera_bias)
        raw_variance = self.variance_head(conditioned_feature)

        if target_size is None:
            target_size = candidate_depth.shape[-2:]

        candidate_depth = F.interpolate(candidate_depth, size=target_size, mode="bilinear", align_corners=False)
        camera_bias = F.interpolate(camera_bias, size=target_size, mode="bilinear", align_corners=False)
        raw_variance = F.interpolate(raw_variance, size=target_size, mode="bilinear", align_corners=False)

        variance_floor = torch.exp(raw_variance.new_tensor(self.min_log_variance))
        variance = variance_floor + F.softplus(raw_variance)
        if self.max_log_variance is not None:
            variance = variance.clamp_max(torch.exp(raw_variance.new_tensor(self.max_log_variance)))

        log_variance = torch.log(variance.clamp_min(1e-8))
        std = torch.sqrt(variance)

        return {
            "candidate_depth": candidate_depth,
            "corrected_depth": candidate_depth + camera_bias,
            "camera_bias": camera_bias,
            "raw_variance": raw_variance,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }