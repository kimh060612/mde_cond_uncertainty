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


class FiLM2d(nn.Module):
    def __init__(self, cond_dim: int, num_channels: int, hidden_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * num_channels),
        )

        # 초기에는 identity modulation
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.mlp(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return x * (1.0 + gamma) + beta


class ResidualFiLMBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        dilation: int = 1,
        film_hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.film1 = FiLM2d(cond_dim, channels, hidden_dim=film_hidden_dim)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.film2 = FiLM2d(cond_dim, channels, hidden_dim=film_hidden_dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.norm1(x)
        y = self.act(y)
        y = self.conv1(y)
        y = self.film1(y, cond)

        y = self.norm2(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.film2(y, cond)

        return residual + y


class MultiScaleContextBlock(nn.Module):
    """
    uncertainty는 local edge뿐 아니라 주변 context가 중요하므로
    dilation conv로 receptive field를 키움.
    """
    def __init__(self, channels: int):
        super().__init__()

        self.branch1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.fuse = nn.Conv2d(3 * channels, channels, 1)

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)

        y = torch.cat([y1, y2, y3], dim=1)
        y = self.fuse(y)
        y = self.act(self.norm(y))

        return x + y


class LargeConditionedUncertaintyHead(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        in_channels: int = 5,
        width: int = 64,
        num_blocks: int = 6,
        min_log_var: float = -5.0,
        max_log_var: float = 3.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=width),
            nn.GELU(),
        )

        dilations = [1, 1, 2, 2, 4, 1]
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ResidualFiLMBlock(
                    channels=width,
                    cond_dim=cond_dim,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
            )
            if i == num_blocks // 2:
                blocks.append(MultiScaleContextBlock(width))

        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=width),
            nn.GELU(),
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, kernel_size=1),
        )

        # 초기 std가 너무 작아지지 않게 설정
        # bias=-2이면 var≈0.135, std≈0.37m
        nn.init.constant_(self.head[-1].bias, -2.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for block in self.blocks:
            if isinstance(block, ResidualFiLMBlock):
                x = block(x, cond)
            else:
                x = block(x)

        log_var = self.head(x)
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)

        return log_var


class ConditionedGaussianDepthAnythingV2(nn.Module):
    """
    Depth Anything V2 with a FiLM-conditioned uncertainty head.

    The depth prediction path can be fine-tuned through ``mu``. The uncertainty
    head receives detached depth-derived proxies plus the camera/scene condition
    vector through FiLM layers.
    """

    def __init__(
        self,
        model_id: str,
        cond_dim: int,
        cache_dir=None,
        freeze_backbone: bool = False,
        min_log_var: float = -5.0,
        max_log_var: float = 3.0,
        uncertainty_width: int = 64,
        uncertainty_blocks: int = 3,
        uncertainty_dropout: float = 0.05,
    ):
        super().__init__()
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_id, cache_dir=cache_dir)
        self.cond_dim = cond_dim

        self.uncertainty_head = LargeConditionedUncertaintyHead(
            cond_dim=cond_dim,
            in_channels=3,
            width=uncertainty_width,
            num_blocks=uncertainty_blocks,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
            dropout=uncertainty_dropout,
        )

        if freeze_backbone:
            for param in self.depth_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        condition: torch.Tensor,
        target_size,
    ):
        if condition.ndim != 2 or condition.shape[1] != self.cond_dim:
            raise ValueError(
                f"condition must have shape [B, {self.cond_dim}], got {tuple(condition.shape)}"
            )

        condition = condition.to(device=pixel_values.device, dtype=pixel_values.dtype)
        outputs = self.depth_model(pixel_values=pixel_values)
        mu = outputs.predicted_depth

        if mu.ndim == 3:
            mu = mu.unsqueeze(1)

        mu = F.interpolate(
            mu,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
        mu = F.softplus(mu)

        rgb_proxy = F.interpolate(
            pixel_values,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        gray = rgb_proxy.mean(dim=1, keepdim=True)

        grad_x = torch.abs(mu[:, :, :, 1:] - mu[:, :, :, :-1])
        grad_y = torch.abs(mu[:, :, 1:, :] - mu[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        grad_mag = grad_x + grad_y

        uncertainty_input = torch.cat([mu.detach(), gray, grad_mag.detach()], dim=1)
        log_var = self.uncertainty_head(uncertainty_input, condition)

        return {
            "predicted_depth": mu,
            "log_variance": log_var,
            "variance": torch.exp(log_var),
            "std": torch.exp(0.5 * log_var),
        }
