import os

import torch
import tqdm.auto as tqdm
import wandb
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.classification import JaccardIndex


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='../models/best_model.pth', verbose=True):
        """
        patience: сколько эпох ждать улучшения
        min_delta: минимальное изменение loss, чтобы считаться улучшением
        path: куда сохранять лучшую модель
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

        # Создаем папку, если её нет
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'✅ Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_epoch(model, optimizer, criterion, device, train_loader, scaler, scheduler=None, num_classes=40):
    model.train()

    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0).to(device)
    loss_log = []
    pbar = tqdm.tqdm(train_loader, desc="Training Epoch")

    for batch_num, (batch_image, batch_heatmaps) in enumerate(pbar):
        batch_image = batch_image.to(device, non_blocking=True)
        batch_heatmaps = batch_heatmaps.to(device, non_blocking=True)

        # Если таргет [B, 1, H, W], сжимаем до [B, H, W]
        if batch_heatmaps.ndim == 4:
            batch_heatmaps = batch_heatmaps.squeeze(1)

        optimizer.zero_grad(set_to_none=True)

        # === AMP ===
        with autocast(dtype=torch.float16 if device != 'mps' else torch.float32):
            logits = model(batch_image)
            loss = criterion(logits, batch_heatmaps)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # === Логирование ===
        with torch.no_grad():
            loss_val = loss.item()
            loss_log.append(loss_val)

            batch_iou = iou_metric(logits, batch_heatmaps)

            if batch_num % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss_val,
                    "train/batch_IoU": batch_iou.item(),  # .item() чтобы достать число из тензора
                    "train/batch_num": batch_num
                })
                # Обновляем описание прогресс-бара
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "iou": f"{batch_iou.item():.4f}"})

    if scheduler is not None:
        scheduler.step()

    # === Финальный расчет за эпоху ===
    avg_loss = float(np.mean(loss_log))
    avg_iou = iou_metric.compute().item()
    iou_metric.reset()

    wandb.log({
        "train/epoch_loss": avg_loss,
        "train/epoch_IoU": avg_iou
    })

    return avg_loss, avg_iou


def val_epoch(model, criterion, device, val_loader, num_classes=40):
    model.eval()

    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0).to(device)
    loss_log = []
    pbar = tqdm.tqdm(val_loader, desc="Validation Epoch")

    with torch.no_grad():
        for batch_num, (batch_image, batch_heatmaps) in enumerate(pbar):
            batch_image = batch_image.to(device, non_blocking=True)
            batch_heatmaps = batch_heatmaps.to(device, non_blocking=True)

            if batch_heatmaps.ndim == 4:
                batch_heatmaps = batch_heatmaps.squeeze(1)

            # === AMP ===
            with autocast(dtype=torch.float16 if device != 'mps' else torch.float32):
                logits = model(batch_image)
                loss = criterion(logits, batch_heatmaps)

            loss_log.append(loss.item())

            # Обновляем метрику
            batch_iou = iou_metric(logits, batch_heatmaps)

            if batch_num % 10 == 0:
                wandb.log({
                    "val/batch_loss": loss.item(),
                    "val/batch_IoU": batch_iou.item(),
                    "val/val_batch_num": batch_num
                })

    avg_loss = float(np.mean(loss_log))
    avg_iou = iou_metric.compute().item()
    iou_metric.reset()

    wandb.log({
        "val/epoch_loss": avg_loss,
        "val/epoch_IoU": avg_iou
    })

    return avg_loss, avg_iou


def train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, device, n_epoch):
    wandb.watch(model, log="all", log_freq=10)
    scaler = GradScaler()

    early_stopping = EarlyStopping(
        patience=7,  # Ждем 7 эпох
        min_delta=0.001,  # Минимальное улучшение лосса
        path="../models/best_model.pth"
    )

    for epoch in tqdm.trange(n_epoch, desc="Training Progress"):
        train_loss, train_iou = train_epoch(model, optimizer, criterion, device, train_loader, scaler, scheduler)
        val_loss, val_iou = val_epoch(model, criterion, device, val_loader)

        # === Проверка Early Stopping ===
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("🛑 Ранняя остановка! Модель перестала обучаться.")
            break

        # === Сохранение чекпоинта ===
        try:
            checkpoint_path = f"../models/checkpoint_{epoch}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_iou": val_iou
            }, checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")


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
