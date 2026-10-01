from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

from torch.utils.data import Dataset
import torch.nn.functional as F


CLASS_NAMES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial Buildings",
    "Pasture",
    "Permanent Crop",
    "Residential Buildings",
    "River",
    "SeaLake",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

NUM_CLASSES = len(CLASS_NAMES)
NUM_BANDS = 13


def read_geotiff(path):
    """
    Read a 13-band EuroSAT GeoTIFF.

    Returns:
        torch.Tensor with shape [13, H, W]
    """

    path = str(path)

    with rasterio.open(path) as src:

        image = src.read()

    image = image.astype(np.float32)

    image = torch.from_numpy(image)

    return image


def resize_multispectral(
    image,
    size=224
):
    """
    Resize a multispectral image.

    Input:
        [C, H, W]

    Output:
        [C, size, size]
    """

    image = image.unsqueeze(0)

    image = F.interpolate(
        image,
        size=(size, size),
        mode="bilinear",
        align_corners=False
    )

    return image.squeeze(0)


def normalize_image(
    image,
    mean,
    std
):
    """
    Normalize each spectral band.

    image:
        [C, H, W]

    mean:
        [C]

    std:
        [C]
    """

    mean = torch.tensor(
        mean,
        dtype=image.dtype
    ).view(
        -1,
        1,
        1
    )

    std = torch.tensor(
        std,
        dtype=image.dtype
    ).view(
        -1,
        1,
        1
    )

    return (
        image - mean
    ) / (
        std + 1e-8
    )


class EuroSATDataset(Dataset):

    def __init__(
        self,
        dataframe,
        mean=None,
        std=None,
        image_size=224,
        augment=False
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.mean = mean
        self.std = std

        self.image_size = image_size
        self.augment = augment

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(
        self,
        index
    ):

        row = self.dataframe.iloc[index]

        image_path = row["image_path"]

        label = int(
            row["label"]
        )

        image = read_geotiff(
            image_path
        )

        # -------------------------------------------------
        # Validate number of spectral bands
        # -------------------------------------------------

        if image.shape[0] != NUM_BANDS:

            raise ValueError(
                f"Expected {NUM_BANDS} bands but "
                f"found {image.shape[0]} in {image_path}"
            )

        # -------------------------------------------------
        # Convert to float
        # -------------------------------------------------

        image = image.float()

        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        image = resize_multispectral(
            image,
            self.image_size
        )

        # -------------------------------------------------
        # Data augmentation
        #
        # Only applied to training data.
        # -------------------------------------------------

        if self.augment:

            # Horizontal flip
            if torch.rand(1).item() < 0.5:

                image = torch.flip(
                    image,
                    dims=[2]
                )

            # Vertical flip
            if torch.rand(1).item() < 0.5:

                image = torch.flip(
                    image,
                    dims=[1]
                )

            # Random 90-degree rotation
            rotations = torch.randint(
                low=0,
                high=4,
                size=(1,)
            ).item()

            if rotations > 0:

                image = torch.rot90(
                    image,
                    rotations,
                    dims=[1, 2]
                )

        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        if self.mean is not None and self.std is not None:

            image = normalize_image(
                image,
                self.mean,
                self.std
            )

        return image, label