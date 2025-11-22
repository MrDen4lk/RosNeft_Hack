import warnings
warnings.filterwarnings("ignore")

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

import dataset
import train_utils
from model import UnetPlusPlusEffNet, DiceCE, UnetSegFormer


def main() -> None:
    # === Аугментация ===

    train_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # === DataLoaders ===

    train_dataset, val_dataset = dataset.create_train_val_datasets(
        data_root="../data",
        val_size=0.2,
        random_state=42,
        train_transforms=train_transform,
        val_transforms=val_transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True
    )

    # === Проверка батча ===

    # images, masks = next(iter(train_loader))
    # print(masks[0, 300:310, 300:310])
    #
    # img = images[0].detach().cpu().permute(1, 2, 0).numpy()
    # mask = masks[0].detach().cpu().numpy()
    #
    # if mask.ndim == 3:  # (1, H, W) -> (H, W)
    #     mask = mask[0]
    #
    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(1, 3, 1)
    # plt.axis("off")
    # plt.imshow(img)
    # plt.title("Input")
    #
    # plt.subplot(1, 3, 2)
    # plt.axis("off")
    # plt.imshow(mask, cmap="gray")
    # plt.title("Mask")
    #
    # plt.subplot(1, 3, 3)
    # plt.axis("off")
    # plt.imshow(img)
    # plt.imshow(mask, cmap="jet", alpha=0.4)
    # plt.title("Input + Mask")
    #
    # plt.tight_layout()
    # plt.show()
    # return

    # === Конфигурация обучения ===

    config = {
        "project": "rosneft-segmentation",
        "experiment": "unet++&effnet_b0",
        "epochs": 30,
        "lr": 3e-4,
        "optimizer": "AdamW",
        "criterion": "Dice&CE",
        "scheduler": "CosineAnnealingWarmRestarts",
        "model": "unet++&efficientnet_bo",
        "device": (
            'cuda' if torch.cuda.is_available()
            else ('mps' if torch.mps.is_available() else 'cpu')
        )
    }

    model = UnetPlusPlusEffNet(num_classes=40)
    model = model.to(config["device"], non_blocking=True)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.encoder.parameters(),
                "lr": config["lr"] * 0.1,
            },
            {
                "params": model.backbone.decoder.parameters(),
                "lr": config["lr"],
            },
            {
                "params": model.backbone.segmentation_head.parameters(),
                "lr": config["lr"],
            },
        ],
        weight_decay=1e-2,
    )

    criterion = DiceCE(ce_weight=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=1,
        eta_min=1e-6
    )

    # === Обучение ===

    wandb.init(project=config["project"], config=config)
    train_utils.train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        n_epoch=config["epochs"]
    )


if __name__ == "__main__":
    main()
