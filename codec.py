# codec.py

from utils import *

# Numerical computing and signal processing
import numpy as np
from scipy.fftpack import dct

# Timing utilities for runtime benchmarking
from time import perf_counter    


def encode_decode_baseline(I, QF):
    """
    Baseline JPEG implementation using the orthonormal 8×8 DCT,
    uniform quantization, and inverse transform. The function returns
    the decoded image, data in spectral domain, and wall-clock runtime.
    """

    # Define C and Q
    ti = perf_counter()
    Q = quantization_matrix(QF)
    C = dct(np.eye(8), axis=0, norm="ortho")
    h, w = I.shape
    hb, wb = h // 8, w // 8
    # Replicate the transform and quantization matrix for each block row
    C = np.broadcast_to(C, (hb, 8, 8))
    Q = np.broadcast_to(Q, (hb, 8, 8))
    # Precompute transposes
    Ct = np.transpose(C, (0, 2, 1))
    # Add singleton dimension for broadcasting
    C = C[:, None]
    Ct = Ct[:, None]
    Q = Q[:, None]
    # Image partitioning
    I_i = umount(I, (8, 8)).reshape(hb, wb, 8, 8)
    # Forward transform
    H_i = (C @ I_i) @ Ct
    # JPEG-like quantization/dequantization
    Hhat_i = np.rint(H_i / Q)
    Htilde_i = Hhat_i * Q
    # Inverse transform
    Ihat_i = (Ct @ Htilde_i) @ C
    # Image reconstruction
    Ihat = remount(Ihat_i.reshape(-1, 8, 8), (h, w))
    tf = perf_counter()
    return Ihat, Htilde_i.reshape(h, w), tf - ti


def encode_decode_proposed(I, QF):
    """
    Proposed JPEG-like implementation using pruned orthonormal DCT,
    uniform quantization, and inverse transform. The function returns
    the decoded image, data in spectral domain, and wall-clock runtime.
    """   
    # Define C and Q
    ti = perf_counter()
    Q = quantization_matrix(QF)
    C = dct(np.eye(8), axis=0, norm="ortho")
    # Compute the number of retained basis vectors for each latitude
    Kstar = determine_kstar()
    # Adjust the transform according to the block-row latitude.
    CK = np.stack([np.vstack((C[:K], np.zeros((8-K, 8)))) for K in Kstar])
    # Decoder always uses the complete orthogonal transform
    C = np.broadcast_to(C, CK.shape)
    # Use the same JPEG quantization matrix for every block row
    Q  = np.broadcast_to(Q, CK.shape)
    # Precompute transposes
    Ct = np.transpose(C, (0, 2, 1))
    CKt = np.transpose(CK, (0, 2, 1))
    # Add singleton dimension for broadcasting
    C   = C[:, None]
    Ct  = Ct[:, None]
    CKt = CKt[:, None]
    Q   = Q[:, None]
    # Image partitioning
    h, w = I.shape
    hb, wb = h // 8, w // 8
    I_i = umount(I, (8, 8)).reshape(hb, wb, 8, 8)
    # Forward transform 
    H_i = (C @ I_i) @ CKt
    # JPEG-like quantization/dequantization
    Hhat_i = np.rint(H_i / Q)
    Htilde_i = Hhat_i * Q
    # Inverse transform 
    Ihat_i = (Ct @ Htilde_i) @ C
    # Image reconstruction
    Ihat = remount(Ihat_i.reshape(-1, 8, 8), (h, w))
    tf = perf_counter()
    return Ihat, Htilde_i.reshape(h, w), tf-ti


