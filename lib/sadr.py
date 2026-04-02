import torch
import torch.nn as nn

from lib.sca_modules import LayerNorm2d


class SizeRoutingClassifier(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial_attention = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self.spatial_attention(x)
        pooled = self.pool(x * attention).flatten(1)
        logits = self.classifier(pooled).view(x.shape[0], 1)
        probabilities = torch.sigmoid(logits)
        return logits, probabilities


class ReceptiveFieldExpert(nn.Module):
    def __init__(self, channels: int, mode: str):
        super().__init__()
        activation = nn.GELU

        if mode == "large":
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                LayerNorm2d(channels),
                activation(),
                nn.Conv2d(channels, channels, 1, bias=False),
            )
        elif mode == "small":
            hidden_channels = channels * 2
            self.block = nn.Sequential(
                nn.Conv2d(channels, hidden_channels, 1, bias=False),
                LayerNorm2d(hidden_channels),
                activation(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
                LayerNorm2d(hidden_channels),
                activation(),
                nn.Conv2d(hidden_channels, channels, 1, bias=False),
                LayerNorm2d(channels),
            )
        else:
            raise ValueError(f"Unsupported expert mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SizeAdaptiveDynamicRouter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.classifier = SizeRoutingClassifier(channels)
        self.small_scale_expert = ReceptiveFieldExpert(channels, mode="small")
        self.large_scale_expert = ReceptiveFieldExpert(channels, mode="large")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, probabilities = self.classifier(x)
        small_feature = self.small_scale_expert(x)
        large_feature = self.large_scale_expert(x)
        routing_weight = probabilities.view(-1, 1, 1, 1)
        routed_feature = routing_weight * large_feature + (1.0 - routing_weight) * small_feature
        return routed_feature, logits
