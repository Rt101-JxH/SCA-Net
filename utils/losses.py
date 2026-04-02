import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, reduction="mean")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = prediction.size(0)
        prediction = prediction.view(batch_size, -1)
        target = target.view(batch_size, -1)
        return self.criterion(prediction, target)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1.0
        batch_size = prediction.size(0)
        prediction = prediction.view(batch_size, -1)
        target = target.view(batch_size, -1)
        intersection = prediction * target
        dice = (2.0 * intersection.sum(dim=1) + smooth) / (
            prediction.sum(dim=1) + target.sum(dim=1) + smooth
        )
        return 1.0 - dice.sum() / batch_size


class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        prediction = torch.sigmoid(prediction)
        prediction = prediction.view(-1)
        target = target.view(-1)
        intersection = (prediction * target).sum()
        total = (prediction + target).sum()
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        return 1.0 - iou


class BCEWithDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.bce = BCELoss(weight)
        self.dice = DiceLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(prediction, target)
        dice_loss = self.dice(torch.sigmoid(prediction), target)
        return bce_loss + dice_loss


class BCEWithIoULoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.bce = BCELoss(weight)
        self.iou = IoULoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(prediction, target) + self.iou(prediction, target)


class StructureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + 5.0 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        weighted_bce = F.binary_cross_entropy_with_logits(prediction, mask, reduction="none")
        weighted_bce = (weight * weighted_bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        prediction = torch.sigmoid(prediction)
        intersection = ((prediction * mask) * weight).sum(dim=(2, 3))
        union = ((prediction + mask) * weight).sum(dim=(2, 3))
        weighted_iou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)

        return (weighted_bce + weighted_iou).mean()


class UncertaintyAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(prediction)
        uncertainty = 4.0 * probability * (1.0 - probability)
        return uncertainty.mean()


class HybridSegmentationLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        focal_gamma: float = 2.0,
        beta: float = 0.5,
        xi: float = 1.35,
        tversky_delta: float = 0.55,
        tversky_gamma: float = 1.2,
        omega: float = 2.0,
        kernel_size: int = 31,
        lambda_geo: float = 0.0,
        lambdas: tuple[float, float] = (2.0, 1.0),
    ):
        super().__init__()
        self.alpha = alpha
        self.focal_gamma = focal_gamma
        self.beta = beta
        self.xi = xi
        self.tversky_delta = tversky_delta
        self.tversky_gamma = tversky_gamma
        self.omega = omega
        self.kernel_size = kernel_size
        self.lambda_geo = lambda_geo
        self.lambdas = lambdas
        self.eps = 1e-6

    def forward(self, prediction: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        prediction = prediction.float()
        mask = mask.float()

        probability = torch.sigmoid(prediction)
        probability = torch.clamp(probability, min=self.eps, max=1.0 - self.eps)

        bce = F.binary_cross_entropy_with_logits(prediction, mask, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = torch.where(mask == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * (1.0 - pt) ** self.focal_gamma * bce

        entropy = -(probability * torch.log2(probability) + (1.0 - probability) * torch.log2(1.0 - probability))
        uncertainty_loss = self.beta * (entropy**self.xi)
        pixel_loss = (focal_loss + uncertainty_loss).mean()

        avg_mask = F.avg_pool2d(
            mask,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        weight_map = 1.0 + self.omega * torch.abs(avg_mask - mask)

        true_positive = (probability * mask * weight_map).sum(dim=(2, 3))
        false_positive = (probability * (1.0 - mask) * weight_map).sum(dim=(2, 3))
        false_negative = ((1.0 - probability) * mask * weight_map).sum(dim=(2, 3))
        tversky = (true_positive + self.eps) / (
            true_positive + self.tversky_delta * false_positive + (1.0 - self.tversky_delta) * false_negative + self.eps
        )
        structure_loss = ((1.0 - tversky) ** self.tversky_gamma).mean()

        geometric_loss = torch.tensor(0.0, device=prediction.device)
        if self.lambda_geo > 0:
            area_prediction = probability.sum(dim=(2, 3))
            area_target = mask.sum(dim=(2, 3))
            ratio_diff = (area_prediction - area_target) / (area_target + 1.0)
            ratio_diff = torch.clamp(ratio_diff, min=-1.0, max=5.0)
            geometric_loss = ratio_diff + F.softplus(-2.0 * ratio_diff) - 0.693147
            geometric_loss = geometric_loss.mean()

        total_loss = self.lambdas[0] * pixel_loss + self.lambdas[1] * structure_loss + self.lambda_geo * geometric_loss
        return total_loss, (pixel_loss, structure_loss, geometric_loss)


class DeepSupervisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_criterion = HybridSegmentationLoss()
        self.aux_criterion = BCEWithDiceLoss()

    def forward(
        self,
        predictions: tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        prediction0, prediction1, prediction2, prediction3, prediction4 = predictions

        loss0, components = self.main_criterion(prediction0, target)
        target_half = F.interpolate(target, scale_factor=0.5, mode="bilinear", align_corners=True)
        loss1 = self.aux_criterion(prediction1, target_half)
        target_quarter = F.interpolate(target, scale_factor=0.25, mode="bilinear", align_corners=True)
        loss2 = self.aux_criterion(prediction2, target_quarter)
        target_eighth = F.interpolate(target, scale_factor=0.125, mode="bilinear", align_corners=True)
        loss3 = self.aux_criterion(prediction3, target_eighth)
        target_sixteenth = F.interpolate(target, scale_factor=0.0625, mode="bilinear", align_corners=True)
        loss4 = self.aux_criterion(prediction4, target_sixteenth)

        return loss0, loss1, loss2, loss3, loss4, components
