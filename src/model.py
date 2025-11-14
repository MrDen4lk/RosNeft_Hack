import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch

class SegModelUnetPlusPlusConvNext(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        logits = self.backbone(x)        # [B, C, H, W]
        return logits


# === Dice + CE для мультикласса ===

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:  [B, C, H, W]
        targets: [B, H, W]   (индексы классов)
                 или [B, 1, H, W]
        """
        num_classes = logits.shape[1]

        # 1) привести форму таргета
        if targets.ndim == 4:          # [B, 1, H, W] -> [B, H, W]
            targets = targets.squeeze(1)

        # 2) привести тип к LongTensor (обязательное требование one_hot)
        targets = targets.long()

        # 3) one-hot: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        probs = torch.softmax(logits, dim=1)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs + targets_one_hot, dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        # targets: [B, H, W] или [B, 1, H, W]
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets_ce = targets.squeeze(1).long()
        else:
            targets_ce = targets.long()

        ce_loss = self.ce(logits, targets_ce)
        dice_loss = self.dice(logits, targets)   # тут внутри уже есть squeeze/long

        return self.ce_weight * ce_loss + (1 - self.ce_weight) * dice_loss