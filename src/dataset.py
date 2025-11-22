from pathlib import Path
from typing import List, Optional, Callable, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from gray_mapping import gray2class


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transforms: Optional[Callable] = None,
    ):
        assert len(image_paths) == len(mask_paths), "images != masks count"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # картинка
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # маска
        mask_gray = Image.open(mask_path).convert("L")
        mask_gray = np.array(mask_gray)

        # в индексы классов
        mask_idx = gray2class[mask_gray]

        if self.transforms is not None:
            aug = self.transforms(image=image, mask=mask_idx)
            image = aug["image"]
            mask_idx = aug["mask"]

        if isinstance(mask_idx, torch.Tensor):
            mask_idx = mask_idx.long()

        return image, mask_idx


def _pair_paths(input_dir: Path, target_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Собирает пары (input, target) по одинаковому имени файла.
    """
    image_paths = sorted(list(input_dir.glob("*.*")))  # png/jpg и т.п.
    paired_imgs = []
    paired_masks = []

    for img_path in image_paths:
        mask_path = target_dir / img_path.name
        if mask_path.exists():
            paired_imgs.append(img_path)
            paired_masks.append(mask_path)
        else:
            print(f"[WARN] mask not found for {img_path.name}, skip")

    return paired_imgs, paired_masks


def create_train_val_datasets(
    data_root: str = "../data",
    val_size: float = 0.2,
    random_state: int = 42,
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable] = None,
) -> Tuple[SegmentationDataset, SegmentationDataset]:
    """
    Создаёт train и val датасеты из папок:
        {data_root}/input
        {data_root}/target
    """
    data_root = Path(data_root)
    input_dir = data_root / "input"
    target_dir = data_root / "target"

    assert input_dir.exists(), f"input dir not found: {input_dir}"
    assert target_dir.exists(), f"target dir not found: {target_dir}"

    images, masks = _pair_paths(input_dir, target_dir)
    assert len(images) > 0, "no image/mask pairs found"

    indices = np.arange(len(images))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=None,
    )

    train_images = [images[i] for i in train_idx]
    train_masks = [masks[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    val_masks = [masks[i] for i in val_idx]

    train_ds = SegmentationDataset(train_images, train_masks, transforms=train_transforms)
    val_ds = SegmentationDataset(val_images, val_masks, transforms=val_transforms)

    return train_ds, val_ds
