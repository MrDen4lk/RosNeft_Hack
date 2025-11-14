import torch
import tqdm.auto as tqdm
import wandb
import numpy as np


# ===== IoU без 0-класса =====

def iou_no_bg(logits: torch.Tensor,
              targets: torch.Tensor,
              num_classes: int,
              eps: float = 1e-6) -> float:
    """
    Считает mean IoU по классам 1..num_classes-1 (0 — фон, игнорируем).

    logits:  [B, C, H, W]
    targets: [B, H, W] или [B, 1, H, W], значения 0..num_classes-1
    """
    # предсказанные классы
    preds = torch.argmax(logits, dim=1)  # [B, H, W]

    # привести таргет к [B, H, W]
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    ious = []

    for c in range(1, num_classes):  # класс 0 (фон) пропускаем
        pred_c = (preds == c)
        targ_c = (targets == c)

        # если класс вообще не встречается ни в предсказании, ни в таргете — пропускаем
        if not targ_c.any() and not pred_c.any():
            continue

        intersection = (pred_c & targ_c).sum().item()
        union = pred_c.sum().item() + targ_c.sum().item() - intersection

        if union == 0:
            continue

        iou_c = intersection / (union + eps)
        ious.append(iou_c)

    if len(ious) == 0:
        return 0.0

    return float(sum(ious) / len(ious))


def train_epoch(model, optimizer, criterion, device, train_loader, scheduler=None, num_classes=40):
    """
    Одна эпоха обучения модели
    """

    loss_log, iou_log = [], []

    model.train()
    for batch_num, (batch_image, batch_heatmaps) in enumerate(tqdm.tqdm(train_loader, desc="Training Epoch")):
        batch_image = batch_image.to(device, non_blocking=True)
        batch_heatmaps = batch_heatmaps.to(device, non_blocking=True)

        # === шаг обучения ===
        optimizer.zero_grad()

        logits = model(batch_image)

        loss = criterion(logits, batch_heatmaps)
        loss.backward()
        optimizer.step()

        # === логирование ===
        with torch.no_grad():
            loss_value = loss.item()

            loss_log.append(loss_value)

            if num_classes is not None:
                batch_iou = iou_no_bg(logits, batch_heatmaps, num_classes=num_classes)
                iou_log.append(batch_iou)
            else:
                batch_iou = None

            if batch_num % 10 == 0:
                log_dict = {
                    "train/batch_loss": loss_value,
                    "train/batch_num": batch_num
                }
                if batch_iou is not None:
                    log_dict["train/batch_IoU"] = batch_iou
                wandb.log(log_dict)

        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    if scheduler is not None:
        scheduler.step()

    avg_loss = float(np.mean(loss_log))
    if iou_log:
        avg_iou = float(np.mean(iou_log))
    else:
        avg_iou = 0.0

    log_dict = {
        "train/epoch_loss": avg_loss,
    }
    if num_classes is not None:
        log_dict["train/epoch_IoU"] = avg_iou

    wandb.log(log_dict)

    return avg_loss, avg_iou


def val_epoch(model, criterion, device, val_loader, num_classes=40):
    """
    Одна эпоха валидации модели
    """

    loss_log, iou_log = [], []

    model.eval()
    for batch_num, (batch_image, batch_heatmaps) in enumerate(tqdm.tqdm(val_loader, desc="Validation Epoch")):
        batch_image = batch_image.to(device, non_blocking=True)
        batch_heatmaps = batch_heatmaps.to(device, non_blocking=True)

        # === предсказание модели ===
        with torch.no_grad():
            logits = model(batch_image)
            loss = criterion(logits, batch_heatmaps)
        loss_log.append(loss.item())

        if num_classes is not None:
            batch_iou = iou_no_bg(logits, batch_heatmaps, num_classes=num_classes)
            iou_log.append(batch_iou)
        else:
            batch_iou = None

        # === логирование ===
        if batch_num % 10 == 0:
            log_dict = {
                "val/batch_loss": loss.item(),
                "val/val_batch_num": batch_num
            }
            if batch_iou is not None:
                log_dict["val/batch_IoU"] = batch_iou
            wandb.log(log_dict)

    avg_loss = float(np.mean(loss_log))
    if iou_log:
        avg_iou = float(np.mean(iou_log))
    else:
        avg_iou = 0.0

    log_dict = {
        "val/epoch_loss": avg_loss,
    }
    if num_classes is not None:
        log_dict["val/epoch_IoU"] = avg_iou

    wandb.log(log_dict)

    return avg_loss, avg_iou


def train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, device, n_epoch):
    """
    Запуск обучения модели
    """

    wandb.watch(model, log="all", log_freq=10)

    for epoch in tqdm.trange(n_epoch, desc="Training Progress"):
        # === обучение и валидация ===
        train_loss = train_epoch(model, optimizer, criterion, device, train_loader, scheduler)
        val_loss = val_epoch(model, criterion, device, val_loader)

        # === логирование ===
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # === сохранение чекпоинта ===
        try:
            model.eval()
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch
            }, f"../models/checkpoint_{epoch}.pth")

            print("✅ Чекпойнт модели сохранён (.pth)")
        except Exception as e:
            print(f"Ошибка при сохранении чекпоинта: {e}")

    # === Сохранение модели через ONNX ===
    try:
        model_cpu = model.to("cpu").eval()
        dummy_input = torch.randn(1, 3, 640, 640, device="cpu")

        onnx_path = "../models/final_model_v1.onnx"
        torch.onnx.export(
            model_cpu,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"✅ ONNX-модель сохранена: {onnx_path}")
    except Exception as e:
        print(f"Ошибка при сохранении модели: {e}")
