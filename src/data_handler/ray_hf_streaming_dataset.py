import torch
import numpy as np
import torchvision.transforms as transforms

import ray
from typing import Dict, List, Tuple
from huggingface_hub import HfFileSystem

DATASET_PATH = "hf://datasets/blanchon/EuroSAT_MSI/data"


# EuroSAT MSI class names mapping
EUROSAT_CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def ray_preprocess(
    batch: Dict[str, np.ndarray],
    list_band_indices: List[int],
) -> Dict[str, np.ndarray]:
    """
    Preprocess a batch of raw parquet images for Ray Data map_batches.
    Converts nested object arrays into proper numpy arrays with shape (C, H, W)
    and selects the specified spectral bands.

    This function is intended to be used with dataset.map_batches() to
    preprocess the raw parquet data before it reaches the training loop.

    ---------
    Arguments
    ---------
    batch: Dict[str, np.ndarray]
        a batch dict from Ray Data with 'image' (object array) and 'label' keys
    list_band_indices: List[int]
        a list of 1-based rasterio band indices to select

    -------
    Returns
    -------
    batch: Dict[str, np.ndarray]
        the batch with 'image' replaced by preprocessed numpy arrays
        of shape (C, H, W) as float32
    """
    zero_based_indices = [i - 1 for i in list_band_indices]

    images = []
    for img in batch["image"]:
        # Handle nested object arrays from parquet format
        if isinstance(img, np.ndarray) and img.dtype == object:
            # Structure: (H,) object -> (W,) object -> (C,) uint16
            msi_array = np.stack(
                [np.stack([np.array(pixel) for pixel in row]) for row in img]
            ).astype(np.float32)
        elif isinstance(img, np.ndarray):
            msi_array = img.astype(np.float32)
        else:
            msi_array = np.array(img, dtype=np.float32)

        # Select bands and convert to (C, H, W)
        if msi_array.ndim == 3 and msi_array.shape[-1] <= 14:
            msi_array = msi_array[:, :, zero_based_indices]
            msi_array = np.transpose(msi_array, (2, 0, 1))
        elif msi_array.ndim == 3:
            msi_array = msi_array[zero_based_indices, :, :]

        images.append(msi_array)

    batch["image"] = np.stack(images)
    return batch


def process_batch(
    batch: Dict[str, np.ndarray],
    is_train: bool = True,
    threshold: int = pow(2, 14),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a preprocessed Ray Data batch into PyTorch tensors.
    Applies normalization and optional training augmentations.

    ---------
    Arguments
    ---------
    batch: Dict[str, np.ndarray]
        a batch dict with 'image' (float32 arrays of shape (C, H, W))
        and 'label' keys, as produced by ray_preprocess
    is_train: bool
        whether to apply training augmentations (default: True)
    threshold: int
        threshold for clamping and normalization (default: 2^14 = 16384)

    -------
    Returns
    -------
    (images_tensor, labels_tensor): Tuple[torch.Tensor, torch.Tensor]
        images_tensor of shape (batch_size, C, H, W) normalized to [0, 1]
        labels_tensor of shape (batch_size,) with integer class labels
    """
    images = batch["image"]
    labels = batch["label"]

    aug_transform = None
    if is_train:
        aug_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

    processed_images = []
    for msi_array in images:
        msi_tensor = torch.from_numpy(msi_array.astype(np.float32))
        if aug_transform is not None:
            msi_tensor = aug_transform(msi_tensor)
        processed_images.append(msi_tensor)

    images_tensor = torch.stack(processed_images)
    images_tensor = torch.clamp(images_tensor, 0, threshold)
    images_tensor = images_tensor / threshold
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)
    return images_tensor, labels_tensor


def get_ray_datasets_4_training(
    dataset_path: str = DATASET_PATH,
) -> Tuple[ray.data.Dataset, ray.data.Dataset]:
    """
    Create Ray Datasets from HuggingFace parquet files.

    ---------
    Arguments
    ---------
    dataset_path: str
        the HuggingFace dataset path (hf:// protocol)

    -------
    Returns
    -------
    (train_ds, val_ds): Tuple[ray.data.Dataset, ray.data.Dataset]
        Ray datasets for training and validation
    """
    hf_fs = HfFileSystem()

    # List all parquet files
    all_files = [f["name"] for f in hf_fs.ls(dataset_path)]

    train_files = [f for f in all_files if "train" in f and f.endswith(".parquet")]
    val_files = [f for f in all_files if "validation" in f and f.endswith(".parquet")]

    # Read parquet files as Ray Datasets using HfFileSystem
    train_ds = ray.data.read_parquet(train_files, filesystem=hf_fs)
    val_ds = ray.data.read_parquet(val_files, filesystem=hf_fs)

    return train_ds, val_ds


def get_ray_datasets_4_testing(
    dataset_path: str = DATASET_PATH,
) -> Tuple[ray.data.Dataset, ray.data.Dataset]:
    """
    Create Ray Datasets for validation and test splits from HuggingFace parquet files.

    ---------
    Arguments
    ---------
    dataset_path: str
        the HuggingFace dataset path (hf:// protocol)

    -------
    Returns
    -------
    (val_ds, test_ds): Tuple[ray.data.Dataset, ray.data.Dataset]
        Ray datasets for validation and testing
    """
    hf_fs = HfFileSystem()

    # List all parquet files
    all_files = [f["name"] for f in hf_fs.ls(dataset_path)]

    val_files = [f for f in all_files if "validation" in f and f.endswith(".parquet")]
    test_files = [f for f in all_files if "test" in f and f.endswith(".parquet")]

    # Read parquet files as Ray Datasets using HfFileSystem
    val_ds = ray.data.read_parquet(val_files, filesystem=hf_fs)
    test_ds = ray.data.read_parquet(test_files, filesystem=hf_fs)

    return val_ds, test_ds
