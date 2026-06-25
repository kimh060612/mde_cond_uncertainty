from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "metric-indoor-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-indoor-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "metric-outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "metric-outdoor-base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
}

class DepthAnythingFiLMUncertainty(nn.Module):
    """
    Hugging Face AutoModelForDepthEstimation 기반 Depth Anything wrapper.

    구조:
        frozen backbone
            -> trainable neck
            -> trainable depth head
            -> predicted depth

        fused decoder feature
            -> uncertainty projection
            -> FiLM conditioning
            -> uncertainty head
            -> log variance

    relative / metric checkpoint 모두 지원:
        AutoModelForDepthEstimation.from_pretrained(model_id)

    Parameters
    ----------
    model_id:
        Hugging Face model ID 또는 로컬 checkpoint 경로.

    context_dim:
        FiLM conditioning vector의 차원.

    cache_dir:
        Hugging Face cache directory.

    uncertainty_channels:
        uncertainty branch의 intermediate channel 수.

    film_hidden_dim:
        context -> gamma, beta MLP hidden dimension.

    uncertainty_feature_index:
        neck이 반환하는 multi-scale feature 중 uncertainty에 사용할 index.
        None이면 config.head_in_index를 사용.

    detach_uncertainty_feature:
        True이면 uncertainty loss가 neck/backbone 쪽으로 전달되지 않음.
        backbone은 어차피 frozen이므로 실제로는 neck gradient만 차단됨.

    min_log_variance, max_log_variance:
        numerical stability를 위한 log variance 범위.

    trust_remote_code:
        custom Hugging Face repository를 사용할 경우 필요할 수 있음.

    torch_dtype:
        torch.float32, torch.float16, torch.bfloat16 등.
    """

    def __init__(
        self,
        model_id: str,
        context_dim: int,
        cache_dir: Optional[str] = None,
        uncertainty_channels: int = 64,
        film_hidden_dim: int = 128,
        uncertainty_feature_index: Optional[int] = None,
        detach_uncertainty_feature: bool = False,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
        trust_remote_code: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        load_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": trust_remote_code,
        }

        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        self.depth_model = (
            AutoModelForDepthEstimation.from_pretrained(
                model_id,
                **load_kwargs,
            )
        )

        self.context_dim = context_dim
        self.detach_uncertainty_feature = detach_uncertainty_feature
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance

        if not hasattr(self.depth_model, "backbone"):
            raise TypeError(
                "Loaded model does not expose `backbone`. "
                "This wrapper expects a Hugging Face Depth Anything-style "
                "depth estimation model."
            )

        if not hasattr(self.depth_model, "neck"):
            raise TypeError(
                "Loaded model does not expose `neck`. "
                "This wrapper expects a Hugging Face Depth Anything-style model."
            )

        if not hasattr(self.depth_model, "head"):
            raise TypeError(
                "Loaded model does not expose `head`. "
                "This wrapper expects a Hugging Face Depth Anything-style model."
            )

        # -------------------------------------------------------------
        # 1. Freeze encoder/backbone
        # -------------------------------------------------------------
        for parameter in self.depth_model.backbone.parameters():
            parameter.requires_grad_(False)

        # -------------------------------------------------------------
        # 2. Keep neck + depth head trainable
        # -------------------------------------------------------------
        for parameter in self.depth_model.neck.parameters():
            parameter.requires_grad_(True)

        for parameter in self.depth_model.head.parameters():
            parameter.requires_grad_(True)

        # Depth head가 사용하는 feature index와 동일하게 두는 것이 기본.
        if uncertainty_feature_index is None:
            uncertainty_feature_index = getattr(
                self.depth_model.config,
                "head_in_index",
                -1,
            )

        self.uncertainty_feature_index = uncertainty_feature_index

        # Depth Anything config에서 neck/fusion channel을 읽음.
        decoder_channels = getattr(
            self.depth_model.config,
            "fusion_hidden_size",
            None,
        )

        if decoder_channels is None:
            raise ValueError(
                "Could not determine decoder feature channels from "
                "`config.fusion_hidden_size`."
            )

        # -------------------------------------------------------------
        # 3. Uncertainty projection
        # -------------------------------------------------------------
        self.uncertainty_projection = nn.Sequential(
            nn.Conv2d(
                decoder_channels,
                uncertainty_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._get_num_groups(uncertainty_channels),
                uncertainty_channels,
            ),
            nn.GELU(),
        )

        # -------------------------------------------------------------
        # 4. FiLM generator
        #
        # context -> gamma, beta
        # FiLM(F, c) = (1 + gamma(c)) * F + beta(c)
        # -------------------------------------------------------------
        self.film_generator = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(
                film_hidden_dim,
                2 * uncertainty_channels,
            ),
        )

        # -------------------------------------------------------------
        # 5. Uncertainty head
        # -------------------------------------------------------------
        hidden_channels = max(uncertainty_channels // 2, 16)

        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(
                uncertainty_channels,
                uncertainty_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._get_num_groups(uncertainty_channels),
                uncertainty_channels,
            ),
            nn.GELU(),

            nn.Conv2d(
                uncertainty_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                self._get_num_groups(hidden_channels),
                hidden_channels,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
            ),
        )

        self._initialize_film_as_identity()

    @staticmethod
    def _get_num_groups(
        channels: int,
        preferred_groups: int = 8,
    ) -> int:
        """
        channels를 나눌 수 있는 가장 큰 GroupNorm group 수를 선택.
        """
        groups = min(channels, preferred_groups)

        while channels % groups != 0:
            groups -= 1

        return groups

    def _initialize_film_as_identity(self) -> None:
        """
        초기에는 FiLM이 feature를 변경하지 않도록 초기화.

        gamma = 0
        beta = 0

        FiLM(F) = (1 + 0) * F + 0 = F
        """
        final_layer = self.film_generator[-1]

        if not isinstance(final_layer, nn.Linear):
            raise TypeError(
                "Unexpected FiLM generator final layer."
            )

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    @property
    def depth_estimation_type(self) -> str:
        """
        현재 checkpoint가 relative인지 metric인지 반환.
        """
        return getattr(
            self.depth_model.config,
            "depth_estimation_type",
            "unknown",
        )

    @property
    def max_depth(self) -> Optional[float]:
        """
        metric checkpoint에서 설정된 maximum depth.
        relative model이면 일반적으로 None.
        """
        return getattr(
            self.depth_model.config,
            "max_depth",
            None,
        )

    def train(
        self,
        mode: bool = True,
    ) -> "DepthAnythingFiLMUncertainty":
        """
        전체 wrapper는 train mode로 두되 frozen backbone은 eval 유지.
        """
        super().train(mode)

        self.depth_model.backbone.eval()
        self.depth_model.neck.train(mode)
        self.depth_model.head.train(mode)

        self.uncertainty_projection.train(mode)
        self.film_generator.train(mode)
        self.uncertainty_head.train(mode)

        return self

    def _extract_backbone_features(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Frozen backbone forward.

        Hugging Face Depth Anything 내부 forward와 동일하게
        forward_with_filtered_kwargs를 사용.
        """
        with torch.no_grad():
            backbone_outputs = (
                self.depth_model.backbone.forward_with_filtered_kwargs(
                    pixel_values,
                    output_hidden_states=False,
                    output_attentions=False,
                )
            )

        return backbone_outputs.feature_maps

    def _forward_decoder(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        backbone -> neck -> depth head를 수동 실행하여
        decoder feature와 predicted depth를 함께 확보.
        """
        feature_maps = self._extract_backbone_features(
            pixel_values
        )

        _, _, height, width = pixel_values.shape

        patch_size = getattr(
            self.depth_model.config,
            "patch_size",
            14,
        )

        if isinstance(patch_size, (list, tuple)):
            patch_height_size = patch_size[0]
            patch_width_size = patch_size[1]
        else:
            patch_height_size = patch_size
            patch_width_size = patch_size

        patch_height = height // patch_height_size
        patch_width = width // patch_width_size

        # Multi-scale decoder features
        decoder_features = self.depth_model.neck(
            feature_maps,
            patch_height,
            patch_width,
        )

        # Original HF depth head
        predicted_depth = self.depth_model.head(
            decoder_features,
            patch_height,
            patch_width,
        )

        return predicted_depth, decoder_features

    def _apply_film(
        self,
        feature: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(
                "context must have shape [B, context_dim], "
                f"but received {tuple(context.shape)}."
            )

        if context.shape[0] != feature.shape[0]:
            raise ValueError(
                "pixel_values and context batch sizes differ: "
                f"{feature.shape[0]} vs {context.shape[0]}."
            )

        if context.shape[1] != self.context_dim:
            raise ValueError(
                f"context dimension must be {self.context_dim}, "
                f"but received {context.shape[1]}."
            )

        film_parameters = self.film_generator(context)
        gamma, beta = film_parameters.chunk(2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return (1.0 + gamma) * feature + beta

    def forward(
        self,
        pixel_values: torch.Tensor,
        context: torch.Tensor,
        target_size: Optional[
            Union[tuple[int, int], torch.Size]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pixel_values:
            AutoImageProcessor로 전처리된 tensor.
            shape: [B, 3, H, W]

        context:
            FiLM conditioning vector.
            shape: [B, context_dim]

        target_size:
            최종 depth/uncertainty output 크기.
            None이면 predicted_depth의 크기를 사용.

            예:
                target_size=(original_height, original_width)

        Returns
        -------
        {
            "predicted_depth": [B, 1, H, W],
            "log_variance":    [B, 1, H, W],
            "variance":        [B, 1, H, W],
            "std":             [B, 1, H, W],
        }
        """
        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape [B, 3, H, W], "
                f"but received {tuple(pixel_values.shape)}."
            )

        predicted_depth, decoder_features = (
            self._forward_decoder(pixel_values)
        )

        # Hugging Face predicted_depth는 일반적으로 [B,H,W]
        if predicted_depth.ndim == 3:
            predicted_depth = predicted_depth.unsqueeze(1)

        uncertainty_feature = decoder_features[
            self.uncertainty_feature_index
        ]

        if uncertainty_feature.ndim != 4:
            raise ValueError(
                "Selected decoder feature must have shape [B,C,H,W], "
                f"but received {tuple(uncertainty_feature.shape)}."
            )

        if self.detach_uncertainty_feature:
            uncertainty_feature = uncertainty_feature.detach()

        uncertainty_feature = self.uncertainty_projection(
            uncertainty_feature
        )

        uncertainty_feature = self._apply_film(
            feature=uncertainty_feature,
            context=context,
        )

        log_variance = self.uncertainty_head(
            uncertainty_feature
        )

        if target_size is None:
            target_size = predicted_depth.shape[-2:]

        predicted_depth = F.interpolate(
            predicted_depth,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        log_variance = F.interpolate(
            log_variance,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        log_variance = log_variance.clamp(
            min=self.min_log_variance,
            max=self.max_log_variance,
        )

        return {
            "predicted_depth": predicted_depth,
            "log_variance": log_variance,
            "variance": torch.exp(log_variance),
            "std": torch.exp(0.5 * log_variance),
        }