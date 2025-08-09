import os
import torch
import numpy as np
import rasterio as rio
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from typing import List, Tuple


from data_handler.data_bands import get_band_indices


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

    def preprocess_msi_image(self, msi_image: torch.Tensor) -> torch.Tensor:
        msi_image = torch.clamp(msi_image, 0, pow(2, 14))
        msi_image = msi_image / pow(2, 14)
        return msi_image

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        file_msi_raster = self.list_images[idx]
        fd_msi_raster = rio.open(file_msi_raster)
        msi_image = fd_msi_raster.read(self.list_band_indices)
        msi_image = torch.from_numpy(msi_image.astype(np.float32))
        if self.is_train_set:
            msi_image = self.transform(msi_image)
        msi_image = self.preprocess_msi_image(msi_image)

        label = self.list_labels[idx]
        return msi_image, label


def split_dataset(
    list_images: List[str],
    list_labels: List[int],
    random_state: int = 29,
    validation_size: float = 0.2,
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
    validation_size: float
        size of validation set (default: 0.2)

    -------
    Returns
    -------
    (list_train_imgs, list_validation_imgs, list_train_lbls, list_validation_lbls): Tuple[List[str], List[str], List[int], List[int]]
        a n-tuple of training and validation image files and their corresponding labels
    """
    list_train_imgs, list_validation_imgs, list_train_lbls, list_validation_lbls = (
        train_test_split(
            list_images,
            list_labels,
            test_size=validation_size,
            random_state=random_state,
        )
    )
    return list_train_imgs, list_validation_imgs, list_train_lbls, list_validation_lbls


def get_dataloaders_for_training(
    list_images: List[str],
    list_labels: List[int],
    list_bands: List[str],
    validation_size: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 8,
    random_state: int = 29,
) -> Tuple[DataLoader, DataLoader]:
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
    validation_size: float
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
    (train_loader, validation_loader): Tuple[DataLoader, DataLoader]
        a tuple of objects for training and validation dataset loaders
    """
    list_train_imgs, list_validation_imgs, list_train_lbls, list_validation_lbls = (
        split_dataset(
            list_images,
            list_labels,
            validation_size=validation_size,
            random_state=random_state,
        )
    )

    train_dataset = EuroSATDataset(
        list_train_imgs, list_train_lbls, list_bands, is_train_set=True
    )
    validation_dataset = EuroSATDataset(
        list_validation_imgs, list_validation_lbls, list_bands, is_train_set=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader


def get_dataloader_for_testing(
    list_images: List[str],
    list_labels: List[int],
    list_bands: List[str],
    validation_size: float = 0.2,
    batch_size: int = 1,
    num_workers: int = 8,
    random_state: int = 29,
) -> DataLoader:
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
    validation_size: float
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
        validation_size=validation_size,
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
    )
    return test_loader
