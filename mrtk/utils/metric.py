import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch


def norm_abs(img: np.array) -> np.array:
    return np.abs(img) / np.max(np.abs(img))


def ssim(img_test: np.array, img_ref: np.array) -> float:
    metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    if img_test.ndim == 2:
        img_test = img_test[np.newaxis, np.newaxis, ...]
    elif img_test.ndim == 3:
        img_test = img_test[np.newaxis, np.newaxis, ...]
    elif img_test.ndim == 4:
        img_test = img_test[np.newaxis, ...]

    if img_ref.ndim == 2:
        img_ref = img_ref[np.newaxis, np.newaxis, ...]
    elif img_ref.ndim == 3:
        img_ref = img_ref[np.newaxis, np.newaxis, ...]
    elif img_ref.ndim == 4:
        img_ref = img_ref[np.newaxis, ...]


    return metric(torch.from_numpy(norm_abs(img_test)), torch.from_numpy(norm_abs(img_ref))).item()


def psnr(img_test: np.array, img_ref: np.array) -> float:
    if img_test.ndim == 2:
        img_test = img_test[np.newaxis, np.newaxis, ...]
    elif img_test.ndim == 3:
        img_test = img_test[np.newaxis, np.newaxis, ...]
    elif img_test.ndim == 4:
        img_test = img_test[np.newaxis, ...]

    if img_ref.ndim == 2:
        img_ref = img_ref[np.newaxis, np.newaxis, ...]
    elif img_ref.ndim == 3:
        img_ref = img_ref[np.newaxis, np.newaxis, ...]
    elif img_ref.ndim == 4:
        img_ref = img_ref[np.newaxis, ...]


    metric = PeakSignalNoiseRatio(data_range=1.0)
    return metric(torch.from_numpy(norm_abs(img_test)), torch.from_numpy(norm_abs(img_ref))).item()