# train_da2_gaussian_nll.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForDepthEstimation

MODEL_IDS = {
    # relative depth
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",

    # metric depth, indoor
    "metric-indoor-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-indoor-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "metric-indoor-large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",

    # metric depth, outdoor
    "metric-outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "metric-outdoor-base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "metric-outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


def _num_groups(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ResidualCNNConnectionNet(nn.Module):
    """
    Spatial size를 유지하는 CNN residual block.

    Batch size가 작아도 안정적으로 동작하도록 BatchNorm 대신 GroupNorm을 사용한다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        dropout: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or out_channels
        padding = dilation

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_num_groups(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
        )
        self.out_act = nn.GELU()

        # Residual branch를 거의 identity에서 시작시켜 초반 log-var 학습을 안정화한다.
        nn.init.constant_(self.residual[-1].weight, 0.0)
        nn.init.constant_(self.residual[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.shortcut(x) + self.residual(x))


class UncertaintyProxyNetwork(nn.Module):
    """
    DA-v2의 predicted depth/RGB/edge proxy를 입력으로 받아서 log variance를 출력한다.
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        num_residual_blocks: int = 4,
        dropout: float = 0.05,
        min_log_var: float = -10.0,
        max_log_var: float = 5.0,
    ):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(width), width),
            nn.GELU(),
        )

        dilations = (1, 2, 1, 4)
        self.residual_body = nn.Sequential(
            *[
                ResidualCNNConnectionNet(
                    in_channels=width,
                    out_channels=width,
                    hidden_channels=width,
                    dropout=dropout,
                    dilation=dilations[idx % len(dilations)],
                )
                for idx in range(num_residual_blocks)
            ]
        )

        mid_channels = max(width // 2, 16)
        self.output_head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(width), width),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(width, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

        # 초기에는 sigma가 너무 작아져 loss가 폭발하지 않도록 약간 큰 variance에서 시작
        nn.init.normal_(self.output_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.output_head[-1].bias, -2.0)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        features = self.input_projection(mu)
        features = self.residual_body(features)
        log_var = self.output_head(features)
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        return log_var


class GaussianDepthAnythingV2(nn.Module):
    """
    Depth Anything v2:
      output predicted_depth -> mean depth mu

    Added uncertainty head:
      input: mu + RGB/luminance proxy
      output: log_var = log sigma^2

    이 방식은 DA-v2 내부 decoder 구조를 직접 뜯지 않고,
    HF AutoModelForDepthEstimation 위에 uncertainty head를 붙이는 안전한 구현이다.
    """

    def __init__(
        self,
        model_id: str,
        freeze_backbone: bool = False,
        min_log_var: float = -10.0,
        max_log_var: float = 5.0,
        uncertainty_width: int = 64,
        uncertainty_blocks: int = 2,
        uncertainty_dropout: float = 0.05,
    ):
        super().__init__()
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        # 입력 채널:
        # 1) predicted depth mu
        # 2) RGB에서 얻은 grayscale intensity
        # 3) depth edge를 표현하는 mu gradient magnitude
        self.uncertainty_head = UncertaintyProxyNetwork(
            in_channels=3,
            width=uncertainty_width,
            num_residual_blocks=uncertainty_blocks,
            dropout=uncertainty_dropout,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
        )

        if freeze_backbone:
            for p in self.depth_model.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor, target_size: Tuple[int, int]):
        outputs = self.depth_model(pixel_values=pixel_values)
        mu = outputs.predicted_depth  # [B, h, w] 또는 [B, H, W]

        if mu.ndim == 3:
            mu = mu.unsqueeze(1)

        mu = F.interpolate(
            mu,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )

        # depth는 양수여야 하므로 안정화.
        # metric model이면 raw가 이미 metric depth에 가까움.
        # relative model을 metric GT에 직접 맞추는 경우도 fine-tuning으로 scale이 학습됨.
        mu = F.softplus(mu)

        # pixel_values는 processor normalized tensor라서 단순 grayscale proxy로만 사용.
        rgb_proxy = F.interpolate(
            pixel_values,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        gray = rgb_proxy.mean(dim=1, keepdim=True)

        # depth edge proxy: 불확실성이 boundary에서 커지는 경향을 학습하기 쉽게 넣음.
        grad_x = torch.abs(mu[:, :, :, 1:] - mu[:, :, :, :-1])
        grad_y = torch.abs(mu[:, :, 1:, :] - mu[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        grad_mag = grad_x + grad_y

        u_in = torch.cat([mu.detach(), gray, grad_mag.detach()], dim=1)
        log_var = self.uncertainty_head(u_in)

        return {
            "mu": mu,
            "log_var": log_var,
            "var": torch.exp(log_var),
            "std": torch.exp(0.5 * log_var),
        }
