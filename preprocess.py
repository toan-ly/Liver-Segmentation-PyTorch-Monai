import os
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
import dicom2nifti
import nibabel as nib
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism, first


def prepare(in_dir,
            pixdim=(1.5, 1.5, 1.0),
            a_min=-200,
            a_max=200,
            spatial_size=(128, 128, 64),
            cache=True
            ):
    """
    Preprocess medical images for deep learning using MONAI.

    Parameters:
        in_dir (str): Input directory containing the image files.
        pixdim (tuple): Desired voxel spacing for resampling the images.
        a_min (int): Minimum intensity value for scaling.
        a_max (int): Maximum intensity value for scaling.
        spatial_size (list): Desired spatial size for resizing the images.
        cache (bool): Flag to cache the dataset in memory.

    Returns:
        train_loader, test_loader: DataLoaders for training and testing datasets.
    """

    set_determinism(seed=0)

    # Define file paths
    path_train_volumes = sorted(
        glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(
        glob(os.path.join(in_dir, "TrainSegmentation", "*.nii.gz")))

    path_test_volumes = sorted(
        glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
    path_test_segmentation = sorted(
        glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))

    # Create file dictionaries
    train_files = [{"vol": img, "seg": seg}
                   for img, seg in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": img, "seg": seg}
                  for img, seg in zip(path_test_volumes, path_test_segmentation)]

    # Define transforms
    keys = ['vol', 'seg']
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstD(keys=keys),
        Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes='RAS'),
        ScaleIntensityRanged(keys="vol", a_min=a_min,
                             a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key='vol'),
        Resized(keys=keys, spatial_size=spatial_size),
        ToTensord(keys=keys),
    ]

    train_transforms = Compose(transforms)
    test_transforms = Compose(transforms)

    # Create datasets and dataloaders
    if cache:
        train_ds = CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=1.0)
        test_ds = CacheDataset(
            data=test_files, transform=test_transforms, cache_rate=1.0)
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        test_ds = Dataset(data=test_files, transform=test_transforms)

    train_loader = DataLoader(train_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)
    return train_loader, test_loader


