import os
import torch
import numpy as np
import rasterio as rio
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from pathlib import PosixPath
from typing import List, Tuple, Union


from data_handler.data_bands import get_band_indices


def read_data_from_tiff(
    file_msi_raster: PosixPath, list_band_indices: Union[None, List[int]]
) -> np.ndarray:
    msi_image = None
    with rio.open(file_msi_raster) as fd_msi_raster:
        if list_band_indices is not None:
            msi_image = fd_msi_raster.read(list_band_indices)
        else:
            msi_image = fd_msi_raster.read()
    return msi_image


def preprocess_msi_image(msi_image: torch.Tensor, threshold: int=10000) -> torch.Tensor:
    msi_image = torch.clamp(msi_image, 0, threshold)
    msi_image = msi_image / threshold
    return msi_image


class EuroSATDataset(Dataset):
    def __init__(
        self,
        list_images: List[str],
        list_labels: List[int],
        list_bands: List[str],
        is_train_set: bool = True,
    ):
        """
        ---------
        Arguments
        ---------
        list_images: List[str]
            a list of strings indicating image files
        list_labels: List[int]
            a list of labels corresponding to the list of image files
        list_bands: List[str]
            a list of sentinel-2 bands that needs to be used for training
        is_train_set: bool
            indicating whether the instance is for train set or not (default: True)
        """
        self.list_images = list_images
        self.list_labels = list_labels
        self.list_band_indices = get_band_indices(list_bands)
        self.transform = None
        self.is_train_set = is_train_set

        if self.is_train_set:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(90),
                ]
            )

    def __len__(self) -> int:
        return len(self.list_images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        file_msi_raster = self.list_images[idx]
        msi_image = read_data_from_tiff(file_msi_raster, self.list_band_indices)

        msi_image = torch.from_numpy(msi_image.astype(np.float32))
        if self.is_train_set:
            msi_image = self.transform(msi_image)
        msi_image = preprocess_msi_image(msi_image)

        label = self.list_labels[idx]
        return msi_image, label


def split_dataset(
    list_images: List[str],
    list_labels: List[int],
    random_state: int = 29,
    val_size: float = 0.2,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    ---------
    Arguments
    ---------
    list_images: List[str]
        a list of train image files
    list_labels: List[int]
        a list of labels corresponding to train image files
    random_state: int
        random state to be used for split (default: 29)
    val_size: float
        size of validation set (default: 0.2)

    -------
    Returns
    -------
    (list_train_imgs, list_val_imgs, list_train_lbls, list_val_lbls): Tuple[List[str], List[str], List[int], List[int]]
        a n-tuple of training and validation image files and their corresponding labels
    """
    list_train_imgs, list_val_imgs, list_train_lbls, list_val_lbls = train_test_split(
        list_images,
        list_labels,
        test_size=val_size,
        random_state=random_state,
    )
    return list_train_imgs, list_val_imgs, list_train_lbls, list_val_lbls


def get_dataloaders_for_training(
    list_images: List[str],
    list_labels: List[int],
    list_bands: List[str],
    val_size: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 8,
    random_state: int = 29,
) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
    """
    ---------
    Arguments
    ---------
    list_images: List[str]
        a list of train image files
    list_labels: List[int]
        a list of labels corresponding to train image files
    list_bands: List[str]
        a list of sentinel-2 bands that needs to be used for training
    val_size: float
        size of validation set (default: 0.2)
    batch_size: int
        batch size to be used for training and validation (default: 64)
    num_workers: int
        number of workers to be used for data loading (default: 8)
    random_state: int
        random state to be used for split (default: 29)

    -------
    Returns
    -------
    (train_loader, val_loader, list_train_imgs, list_val_imgs): Tuple[DataLoader, DataLoader, List[str], List[str]]
        a tuple of objects for training and validation dataset loaders
    """
    list_train_imgs, list_val_imgs, list_train_lbls, list_val_lbls = split_dataset(
        list_images,
        list_labels,
        val_size=val_size,
        random_state=random_state,
    )

    train_dataset = EuroSATDataset(
        list_train_imgs, list_train_lbls, list_bands, is_train_set=True
    )
    val_dataset = EuroSATDataset(
        list_val_imgs, list_val_lbls, list_bands, is_train_set=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader, list_train_imgs, list_val_imgs


def get_dataloader_for_testing(
    list_images: List[str],
    list_labels: List[int],
    list_bands: List[str],
    val_size: float = 0.2,
    batch_size: int = 1,
    num_workers: int = 8,
    random_state: int = 29,
) -> Tuple[DataLoader, List[str]]:
    """
    ---------
    Arguments
    ---------
    list_images: List[str]
        a list of train image files
    list_labels: List[int]
        a list of labels corresponding to train image files
    list_bands: List[str]
        a list of sentinel-2 bands that needs to be used for testing
    val_size: float
        size of validation set (default: 0.2)
    batch_size: int
        batch size to be used for testing (default: 1)
    num_workers: int
        number of workers to be used for data loading (default: 8)
    random_state: int
        random state to be used for split (default: 29)

    -------
    Returns
    -------
    test_loader: DataLoader
        an object for test dataset loader
    """
    _, list_test_imgs, _, list_test_lbls = split_dataset(
        list_images,
        list_labels,
        val_size=val_size,
        random_state=random_state,
    )

    test_dataset = EuroSATDataset(
        list_test_imgs, list_test_lbls, list_bands, is_train_set=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return test_loader, list_test_imgs
