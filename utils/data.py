from pathlib import Path
import random
from typing import Union

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _collect_files(root: Union[str, Path]) -> dict[str, Path]:
    root_path = Path(root)
    return {
        path.stem: path
        for path in sorted(root_path.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _collect_sorted_paths(root: Union[str, Path]) -> list[Path]:
    root_path = Path(root)
    return [
        path
        for path in sorted(root_path.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _resize_image(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.BILINEAR)


def _resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return mask.resize((size, size), Image.NEAREST)


class PairRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class PairRandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


class PairRandomRotate:
    def __init__(self, degrees: tuple[float, float] = (-180.0, 180.0), p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        return image, mask


class PolypSegmentationDataset(Dataset):
    def __init__(
        self,
        image_root: Union[str, Path],
        mask_root: Union[str, Path],
        image_size: int,
        augment: bool = True,
    ):
        image_files = _collect_files(image_root)
        mask_files = _collect_files(mask_root)
        common_keys = sorted(set(image_files) & set(mask_files))

        self.image_paths = [image_files[key] for key in common_keys]
        self.mask_paths = [mask_files[key] for key in common_keys]
        self.image_size = image_size
        self.augment = augment

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = transforms.ToTensor()
        self.random_rotate = PairRandomRotate()
        self.random_vflip = PairRandomVerticalFlip()
        self.random_hflip = PairRandomHorizontalFlip()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

        image = _resize_image(image, self.image_size)
        mask = _resize_mask(mask, self.image_size)

        if self.augment:
            image, mask = self.random_rotate(image, mask)
            image, mask = self.random_vflip(image, mask)
            image, mask = self.random_hflip(image, mask)

        image_tensor = self.image_transform(image)
        mask_tensor = (self.mask_transform(mask) > 0.5).float()
        return image_tensor, mask_tensor


class InferenceDataset(Dataset):
    def __init__(self, image_root: Union[str, Path], image_size: int, mask_root: Union[str, Path, None] = None):
        self.image_paths = _collect_sorted_paths(image_root)
        self.mask_paths = _collect_sorted_paths(mask_root) if mask_root is not None else None
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        if self.mask_paths is None:
            return len(self.image_paths)
        return min(len(self.image_paths), len(self.mask_paths))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[int, int], str]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.mask_paths is None:
            width, height = image.size
        else:
            mask = Image.open(self.mask_paths[index]).convert("L")
            width, height = mask.size
        image_tensor = self.image_transform(image)
        output_name = f"{image_path.stem}.png"
        return image_tensor, (height, width), output_name


def create_train_loader(
    image_root: Union[str, Path],
    mask_root: Union[str, Path],
    batch_size: int,
    image_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
) -> DataLoader:
    dataset = PolypSegmentationDataset(image_root, mask_root, image_size=image_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
