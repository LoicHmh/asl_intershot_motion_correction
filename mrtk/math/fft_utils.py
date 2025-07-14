import numpy as np


def ifftnd(ksp, axes=[-1], shift=True):
    if axes is None:
        axes = range(ksp.ndim)
    if shift:
        ksp = np.fft.ifftshift(ksp, axes=axes)
    img = np.fft.ifftn(ksp, axes=axes)
    if shift:
        img = np.fft.fftshift(img, axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img


def fftnd(img, axes=[-1], shift=True):
    if axes is None:
        axes = range(img.ndim)
    if shift:
        img = np.fft.ifftshift(img, axes=axes)
    ksp = np.fft.fftn(img, axes=axes)
    if shift:
        ksp = np.fft.fftshift(ksp, axes=axes)
    ksp /= np.sqrt(np.prod(np.take(ksp.shape, axes)))
    return ksp

def rms_comb(sig, axis=1):
    return np.sqrt(np.sum(abs(sig)**2, axis))


def roll(img, r=[1, 1, 1]):
    img = np.roll(img, r[0], axis=0)
    img = np.roll(img, r[1], axis=1)
    img = np.roll(img, r[2], axis=2)
    return img 
