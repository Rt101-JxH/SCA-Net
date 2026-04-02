import torch
import torch.nn as nn
import timm
from torchvision.transforms.functional import rgb_to_grayscale

from lib.sadr import SizeAdaptiveDynamicRouter
from lib.sca_modules import (
    LaplacianGuidedSynergisticRefiner,
    LayerNorm2d,
    SemanticModuleGroup,
    build_laplacian_pyramid,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            LayerNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, in_channels // 4),
            ConvBlock(in_channels // 4, out_channels),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, decoder_feature: torch.Tensor, skip_feature: torch.Tensor) -> torch.Tensor:
        fused_feature = torch.cat([skip_feature, decoder_feature], dim=1)
        return self.upsample(self.block(fused_feature))


class PredictionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = ConvBlock(in_channels, in_channels // 4)
        self.classifier = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.projection(x))


class SCANet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, pretrained_backbone: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            "pvt_v2_b4",
            pretrained=pretrained_backbone,
            features_only=True,
            drop_path_rate=0.3,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            LayerNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.semantic_module_group = SemanticModuleGroup(
            in_channels_list=(64, 128, 320, 512),
            hidden_dim=256,
            num_layers=6,
            pool_size=7,
        )
        self.sadr = SizeAdaptiveDynamicRouter(512)

        self.bridge = nn.Sequential(
            ConvBlock(512, 512),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.decoder4 = DecoderBlock(768, 256)
        self.decoder3 = DecoderBlock(384, 128)
        self.decoder2 = DecoderBlock(192, 64)
        self.decoder1 = DecoderBlock(128, 32)

        self.lgsr4 = LaplacianGuidedSynergisticRefiner(256)
        self.lgsr3 = LaplacianGuidedSynergisticRefiner(128)
        self.lgsr2 = LaplacianGuidedSynergisticRefiner(64)
        self.lgsr1 = LaplacianGuidedSynergisticRefiner(64)

        self.output5 = PredictionHead(512, num_classes)
        self.output4 = PredictionHead(256, num_classes)
        self.output3 = PredictionHead(128, num_classes)
        self.output2 = PredictionHead(64, num_classes)
        self.output1 = PredictionHead(32, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            grayscale = rgb_to_grayscale(x)
            laplacian_pyramid = build_laplacian_pyramid(grayscale, levels=5)
            laplacian_prior = laplacian_pyramid[1]

        stage0 = self.stem(x)
        stage1, stage2, stage3, stage4 = self.backbone(x)
        stage1, stage2, stage3, stage4 = self.semantic_module_group([stage1, stage2, stage3, stage4])
        stage4, routing_logits = self.sadr(stage4)

        decoder5 = self.bridge(stage4)
        output5 = self.output5(decoder5)

        refined4 = self.lgsr4(stage3, output5, laplacian_prior)
        decoder4 = self.decoder4(decoder5, refined4)
        output4 = self.output4(decoder4)

        refined3 = self.lgsr3(stage2, output4, laplacian_prior)
        decoder3 = self.decoder3(decoder4, refined3)
        output3 = self.output3(decoder3)

        refined2 = self.lgsr2(stage1, output3, laplacian_prior)
        decoder2 = self.decoder2(decoder3, refined2)
        output2 = self.output2(decoder2)

        refined1 = self.lgsr1(stage0, output2, laplacian_prior)
        decoder1 = self.decoder1(decoder2, refined1)
        output1 = self.output1(decoder1)

        return output1, output2, output3, output4, output5, routing_logits
