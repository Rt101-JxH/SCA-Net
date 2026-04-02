import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel = torch.tensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    return (kernel / 256.0).repeat(channels, 1, 1, 1)


def _gaussian_blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padded = F.pad(image, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(padded, kernel, groups=image.shape[1])


def _downsample(image: torch.Tensor) -> torch.Tensor:
    return image[:, :, ::2, ::2]


def _upsample(image: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(image)
    up = torch.cat([image, zeros], dim=3)
    up = up.view(image.shape[0], image.shape[1], image.shape[2] * 2, image.shape[3])
    up = up.permute(0, 1, 3, 2)
    up = torch.cat([up, torch.zeros_like(up)], dim=3)
    up = up.view(image.shape[0], image.shape[1], image.shape[3] * 2, image.shape[2] * 2)
    up = up.permute(0, 1, 3, 2)
    kernel = _gaussian_kernel(image.shape[1], image.device, torch.float32)
    return _gaussian_blur(up, 4 * kernel)


def build_laplacian_pyramid(image: torch.Tensor, levels: int) -> list[torch.Tensor]:
    current = image
    pyramid = []

    for _ in range(levels):
        kernel = _gaussian_kernel(current.shape[1], current.device, torch.float32)
        filtered = _gaussian_blur(current, kernel)
        down = _downsample(filtered)
        up = _upsample(down)
        if up.shape[2:] != current.shape[2:]:
            up = F.interpolate(up, size=current.shape[2:])
        pyramid.append(current - up)
        current = down

    pyramid.append(current)
    return pyramid


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding[:, : x.shape[1], :]


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg_out + max_out)


class SelectiveFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.attention = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = self.local_branch(x)
        context_features = self.context_branch(x)
        fused_features = self.fuse(torch.cat([local_features, context_features], dim=1))
        fused_features = self.attention(fused_features)
        return x + fused_features


class CrossScaleGlobalAggregator(nn.Module):
    def __init__(
        self,
        in_channels_list: tuple[int, ...] = (64, 128, 320, 512),
        hidden_dim: int = 256,
        num_layers: int = 6,
        pool_size: int = 7,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        self.num_scales = len(in_channels_list)
        self.projections = nn.ModuleList([nn.Conv2d(channels, hidden_dim, 1) for channels in in_channels_list])
        self.position_encoding = PositionalEncoding(
            hidden_dim, max_len=pool_size * pool_size * self.num_scales
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        batch_size = features[0].shape[0]
        tokens = []

        for projection, feature in zip(self.projections, features):
            token = projection(feature)
            token = F.adaptive_avg_pool2d(token, (self.pool_size, self.pool_size))
            token = token.flatten(2).transpose(1, 2)
            tokens.append(token)

        sequence = torch.cat(tokens, dim=1)
        sequence = self.position_encoding(sequence)
        sequence = self.transformer(sequence)

        semantic_features = []
        for token in torch.chunk(sequence, self.num_scales, dim=1):
            semantic_feature = token.transpose(1, 2).reshape(
                batch_size, self.hidden_dim, self.pool_size, self.pool_size
            )
            semantic_features.append(semantic_feature)

        return semantic_features


class GatedSemanticInjection(nn.Module):
    def __init__(self, local_channels: int, semantic_channels: int = 256):
        super().__init__()
        self.local_projection = nn.Sequential(
            nn.Conv2d(local_channels, local_channels, 1, bias=False),
            LayerNorm2d(local_channels),
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(semantic_channels, local_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.semantic_content = nn.Sequential(
            nn.Conv2d(semantic_channels, local_channels, 1, bias=False),
            LayerNorm2d(local_channels),
        )
        self.smoothing = nn.Sequential(
            nn.Conv2d(local_channels, local_channels, 3, padding=1, groups=local_channels, bias=False),
            LayerNorm2d(local_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, local_feature: torch.Tensor, semantic_feature: torch.Tensor) -> torch.Tensor:
        residual = local_feature
        height, width = local_feature.shape[2:]
        if semantic_feature.shape[2:] != (height, width):
            semantic_feature = F.interpolate(
                semantic_feature,
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )

        local_feature = self.local_projection(local_feature)
        gated_feature = local_feature * self.semantic_gate(semantic_feature)
        content_feature = self.semantic_content(semantic_feature)
        fused_feature = gated_feature + content_feature
        return self.smoothing(fused_feature) + residual


class SemanticModuleGroup(nn.Module):
    def __init__(
        self,
        in_channels_list: tuple[int, ...] = (64, 128, 320, 512),
        hidden_dim: int = 256,
        num_layers: int = 6,
        pool_size: int = 7,
    ):
        super().__init__()
        self.aggregator = CrossScaleGlobalAggregator(
            in_channels_list=in_channels_list,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pool_size=pool_size,
        )
        self.local_adapters = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Sequential(nn.Conv2d(in_channels_list[2], 256, 1), LayerNorm2d(256), nn.ReLU(inplace=True)),
                nn.Identity(),
            ]
        )
        self.injectors = nn.ModuleList(
            [
                GatedSemanticInjection(64, hidden_dim),
                GatedSemanticInjection(128, hidden_dim),
                GatedSemanticInjection(256, hidden_dim),
                GatedSemanticInjection(512, hidden_dim),
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        semantic_features = self.aggregator(features)
        adapted_features = [adapter(feature) for adapter, feature in zip(self.local_adapters, features)]
        return [
            injector(local_feature, semantic_feature)
            for injector, local_feature, semantic_feature in zip(
                self.injectors, adapted_features, semantic_features
            )
        ]


class LaplacianGuidedSynergisticRefiner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, channels // 2, 3, padding=1, bias=False),
            LayerNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        self.feature_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1),
            LayerNorm2d(channels),
            nn.Sigmoid(),
        )
        self.edge_gate = nn.Sequential(
            nn.Conv2d(channels + (channels // 2), channels, 1),
            LayerNorm2d(channels),
            nn.Sigmoid(),
        )
        self.selective_fusion = SelectiveFusion(channels)
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        current_feature: torch.Tensor,
        deeper_prediction: torch.Tensor,
        laplacian_prior: torch.Tensor,
    ) -> torch.Tensor:
        height, width = current_feature.shape[2:]
        refined_feature = self.feature_projection(current_feature)

        if deeper_prediction.shape[2:] != (height, width):
            deeper_prediction = F.interpolate(
                deeper_prediction, size=(height, width), mode="bilinear", align_corners=True
            )
        reverse_semantic_map = 1.0 - torch.sigmoid(deeper_prediction)

        if laplacian_prior.shape[2:] != (height, width):
            laplacian_prior = F.interpolate(
                laplacian_prior, size=(height, width), mode="bilinear", align_corners=True
            )
        edge_feature = self.edge_encoder(laplacian_prior)

        semantic_guidance = self.semantic_gate(torch.cat([refined_feature, reverse_semantic_map], dim=1))
        semantic_feature = refined_feature * semantic_guidance + refined_feature

        edge_guidance = self.edge_gate(torch.cat([semantic_feature, edge_feature], dim=1))
        edge_feature = semantic_feature * edge_guidance
        fused_feature = self.selective_fusion(edge_feature)
        return self.output_projection(fused_feature)
