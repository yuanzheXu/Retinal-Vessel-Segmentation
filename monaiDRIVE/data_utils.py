import json
import numpy as np
import cv2
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    RandCropByPosNegLabeld, RandRotate90d, RandFlipd,
    RandGaussianNoised, RandAdjustContrastd, EnsureTyped, Lambdad,
    Rand2DElasticd, RandAffined, OneOf, RandShiftIntensityd
)
from monai.data import CacheDataset, DataLoader, partition_dataset, PILReader, ITKReader, list_data_collate, Dataset


def binarize_label(x):
    return (x > 0.5).astype(np.float32)


def select_first_channel_label(x):
    return x[:1, :, :] if x.shape[0] > 1 else x

def enhance_fundus(img_numpy):
    if img_numpy.shape[0] == 3:
        green = img_numpy[1, :, :]
    else:
        green = img_numpy[0, :, :]
    green = (green - green.min()) / (green.max() - green.min() + 1e-8) * 255.0
    green = green.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    return np.expand_dims(enhanced.astype(np.float32) / 255.0, axis=0)


def get_transforms(mode="train", spatial_size=(256, 256), num_samples=4):
    if mode == "train":
        return Compose([
            LoadImaged(keys=["image"], reader=ITKReader),
            LoadImaged(keys=["label"], reader=PILReader),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys=["image"], func=enhance_fundus),
            Lambdad(keys=["label"], func=select_first_channel_label),
            Lambdad(keys=["label"], func=binarize_label),
            RandRotate90d(keys=["image", "label"], prob=0.75, spatial_axes=(0, 1)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandAffined(
                keys=["image", "label"],
                prob=0.5,
                rotate_range=np.pi / 12,
                shear_range=np.pi / 18,
                translate_range=(20, 20),
                padding_mode="reflection"
            ),

            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=3, neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0
            ),

            Rand2DElasticd(
                keys=["image", "label"],
                prob=0.5,
                spacing=(30, 30),
                magnitude_range=(2, 6),
                padding_mode="reflection",
                mode=["bilinear", "nearest"]
            ),

            OneOf([
                RandGaussianNoised(keys="image", prob=1.0, mean=0.0, std=0.05),
                RandShiftIntensityd(keys="image", prob=1.0, offsets=0.1),
                RandAdjustContrastd(keys="image", prob=1.0, gamma=(0.5, 1.5))
            ], weights=[0.2, 0.4, 0.4]),

            EnsureTyped(keys=["image", "label"]),
        ])
    else:
        return Compose([
            LoadImaged(keys=["image"], reader=ITKReader),
            LoadImaged(keys=["label"], reader=PILReader),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys=["image"], func=enhance_fundus),
            Lambdad(keys=["label"], func=select_first_channel_label),
            Lambdad(keys=["label"], func=binarize_label),
            EnsureTyped(keys=["image", "label"]),
        ])


def get_dataloaders(json_path, batch_size=1, num_samples=4, fold=0, num_folds=5):
    with open(json_path, 'r') as f:
        data = json.load(f)
    train_files = data['training']

    folds_list = partition_dataset(
        data=train_files,
        num_partitions=num_folds,
        shuffle=True,
        seed=42
    )

    val_ds_list = folds_list[fold]
    train_ds_list = [item for i, f in enumerate(folds_list) if i != fold for item in f]

    train_ds = Dataset(
        data=train_ds_list,
        transform=get_transforms("train", spatial_size=(256, 256), num_samples=num_samples)
    )

    val_ds = CacheDataset(
        data=val_ds_list,
        transform=get_transforms("val"),
        cache_rate=1.0,
        num_workers=0
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader