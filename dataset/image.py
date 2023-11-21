from typing import Callable

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, CIFAR10, CelebA

import os
import random
from PIL import Image
from typing import Tuple

from torchvision import transforms as T


def transform_fn_eval(image_size: Tuple[int, int]) -> T.Compose:
    return T.Compose(
        [   
            T.CenterCrop(140),
            T.Resize(image_size),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1),
        ]
    )

def transform_fn(image_size: Tuple[int, int]) -> T.Compose:
    return T.Compose(

        [   T.CenterCrop(140),
            T.Resize(image_size),
            #T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1),
            
        ]
    )


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        transform: Callable = None,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str = None) -> None:
        self.dataset = ImageFolder(self.data_dir, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    

class CifarDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        transform: Callable = None,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str = None) -> None:
        self.dataset = CIFAR10(self.data_dir, transform=self.transform, download=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    

class CelebaDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        transform: Callable = None,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str = None) -> None:
        self.dataset = CelebA(self.data_dir, transform=self.transform, download=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )



class afhqDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, source: str, target: str, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.target = target
        self.source = source
        self.transform = transform
        self.source_list = sorted(os.listdir(os.path.join(data_dir, source)))
        self.target_list = sorted(os.listdir(os.path.join(data_dir, target)))

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        target_filename = self.target_list[index]
        source_filename = random.choice(self.source_list)
        target_path = os.path.join(self.data_dir, self.target, target_filename)
        source_path = os.path.join(self.data_dir, self.source, source_filename)
        target_image = Image.open(target_path).convert("RGB")
        source_image = Image.open(source_path).convert("RGB")

        if self.transform is not None:
            target_image = self.transform(target_image)
            source_image = self.transform(source_image)

        return {'target':target_image, 'source':source_image}


class afhqDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        source: str,
        target: str,
        transform: Callable = None,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target = target
        self.source = source

    def setup(self, stage: str = None) -> None:
        self.train_dataset = afhqDataset(os.path.join(self.data_dir,'train'), self.source, self.target, transform=self.transform)
        self.val_dataset = afhqDataset(os.path.join(self.data_dir,'val'), self.source, self.target, transform=self.transform)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    














class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, length: int, noise_shape: tuple):
        self.length = length
        self.noise_shape = noise_shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(self.noise_shape)


class NoiseDataModule(LightningDataModule):
    def __init__(
            self,
            length: int,
            noise_shape: tuple,
            batch_size: int = 32,
            num_workers: int = 2,
            pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.length = length
        self.noise_shape = noise_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str = None) -> None:
        self.test_dataset = NoiseDataset(self.length, self.noise_shape)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
