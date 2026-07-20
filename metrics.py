# metrics.py

# Numerical computing and signal processing
import numpy as np
from scipy.signal import convolve2d

def bpp(x):
    """
    Estimate the coding rate (bits/pixel) assuming that only non-zero
    transform coefficients are transmitted. Each non-zero coefficient
    is assumed to require 8 bits.
    """
    x = np.rint(x).ravel()
    return 8 * np.count_nonzero(x) / x.size


def weights(shape):
    """
    Compute the latitude-dependent weights used by weighted spherical
    quality metrics (WS-PSNR and WS-SSIM). The weights compensate for
    the non-uniform sampling density of ERP images.
    """
    h, w = shape
    phi = np.arange(h + 1) * np.pi / h
    col = (2 * np.pi / w) * (-np.cos(phi[1:]) + np.cos(phi[:-1]))
    return np.repeat(col[:, None], w, axis=1)


def wspsnr(x, y, peak=255):
    """
    Compute the Weighted-to-Spherically-uniform PSNR (WS-PSNR) for
    omnidirectional images represented in ERP format. The local mean
    squared error values are averaged using latitude-dependent weights to
    account for the non-uniform sampling density of the ERP projection.
    """
    w = weights(x.shape)
    wmse = np.sum((x - y) ** 2 * w) / (4 * np.pi)
    return 10 * np.log10(peak**2 / wmse)


def wsssim(x, y, K1=0.01, K2=0.03, L=255):
    """
    Compute the Weighted-to-Spherically-uniform SSIM (WS-SSIM) for
    omnidirectional images represented in ERP format. The local SSIM
    values are averaged using latitude-dependent weights to account
    for the non-uniform sampling density of the ERP projection.
    """
    x = x.astype(float)
    y = y.astype(float)

    k = 11
    sigma = 1.5

    t = np.arange(-(k // 2), k // 2 + 1)
    X, Y = np.meshgrid(t, t)
    g = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    g /= g.sum()

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = convolve2d(x, g, mode="valid")
    mu2 = convolve2d(y, g, mode="valid")

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2

    sigma1 = convolve2d(x * x, g, mode="valid") - mu1_sq
    sigma2 = convolve2d(y * y, g, mode="valid") - mu2_sq
    sigma12 = convolve2d(x * y, g, mode="valid") - mu12

    W = weights(x.shape)[5:-5, 5:-5]

    ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1 + sigma2 + C2)
    )

    return np.average(ssim, weights=W)