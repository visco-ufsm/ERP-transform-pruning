# utils.py

# Numerical computing and signal processing
import numpy as np

def umount(image, block_size):
    """
    Split an image into non-overlapping blocks without copying data.
    Output shape: (num_blocks, block_height, block_width).
    """
    nrows, ncols = block_size
    h, w = image.shape
    return (
        image.reshape(h // nrows, nrows, -1, ncols)
             .swapaxes(1, 2)
             .reshape(-1, nrows, ncols)
    )


def remount(blocks, shape):
    """
    Reconstruct an image from a sequence of non-overlapping blocks.
    Inverse operation of `umount`.
    """
    h, w = shape
    _, nrows, ncols = blocks.shape
    return (
        blocks.reshape(h // nrows, -1, nrows, ncols)
              .swapaxes(1, 2)
              .reshape(h, w)
    )


def determine_kstar(h=1920, N=8):
    """
    Determine the number of retained DCT basis vectors (K*) for each
    block row according to its latitude in the ERP image.
    """
    phi = np.pi / 2 - (np.pi / h) * np.linspace(0, h, h // N)
    return np.ceil(N * np.cos(phi)).astype(int)
    


def quantization_matrix(QF):
    """
    Generate the standard JPEG luminance quantization matrix for the
    specified quality factor.
    """
    S = 5000 / QF if QF < 50 else 200 - 2 * QF

    Q0 = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])

    return np.floor((S * Q0 + 50) / 100)
