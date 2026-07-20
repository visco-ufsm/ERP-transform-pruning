"""
Reference implementation of the proposed geometry-aware transform
pruning scheme for ERP image compression.

The script compares the proposed method against a JPEG baseline
using estimated bitrate, WS-PSNR, and WS-SSIM. Reported execution
times correspond to high-level NumPy operations; they should not 
be interpreted as optimized codec runtimes.
"""

# Image I/O
from skimage.io import imread

# Visualization
from matplotlib import pyplot as plt

from metrics import *
from codec import *

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load image and initialize quantization parameters
    # ------------------------------------------------------------

    filename = 'sample/PoleVault_le_3840x1920_30fps_8bit_420_erp_0.bmp'
    I = imread(filename, as_gray=True).astype(np.float32) * 255
    QF = 5                

    # ------------------------------------------------------------
    # Run the JPEG baseline
    # ------------------------------------------------------------

    Itilde_JPEG, spectral_JPEG, time_JPEG = encode_decode_baseline(I, QF)

    # ------------------------------------------------------------
    # Run the proposed method
    # ------------------------------------------------------------
    
    Itilde_prop, spectral_prop, time_prop = encode_decode_proposed(I, QF) 

    # ------------------------------------------------------------
    # Evaluate the JPEG baseline
    # ------------------------------------------------------------

    bpp_JPEG = bpp(spectral_JPEG)
    wsssim_JPEG = wsssim(Itilde_JPEG, I)
    wspsnr_JPEG = wspsnr(Itilde_JPEG, I)
    print('-'*20)
    print('JPEG baseline')
    print('-'*20)
    print('Rate: %.3f bpp' % bpp_JPEG)
    print('WS-PSNR: %.3f dB' % wspsnr_JPEG)
    print('WS-SSIM: %.3f' % wsssim_JPEG)
    print('EET: %.3f s' % time_JPEG)
    print('-'*20)

    # ------------------------------------------------------------
    # Evaluate the proposed method
    # ------------------------------------------------------------

    bpp_prop = bpp(spectral_prop)
    wsssim_prop = wsssim(Itilde_prop, I)
    wspsnr_prop = wspsnr(Itilde_prop, I)
    print('-'*20)
    print('The proposed method')
    print('-'*20)
    print('Rate: %.3f bpp' % bpp_prop)
    print('WS-PSNR: %.3f dB' %  wspsnr_prop)
    print('WS-SSIM: %.3f' % wsssim_prop)
    print('EET: %.3f s' % time_prop)
    print('-'*20)

    # ------------------------------------------------------------
    # Display visual comparison
    # ------------------------------------------------------------

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display each image in its respective subplot
    axes[0].imshow(I, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')  # Hide grid lines and pixel ticks

    axes[1].imshow(Itilde_JPEG, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("JPEG (WS-PSNR: %.2f dB | WS-SSIM: %.2f | Rate: %.2f bpp)" % (wspsnr_JPEG, wsssim_JPEG, bpp_JPEG), fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(Itilde_prop, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title("Proposed (WS-PSNR: %.2f dB | WS-SSIM: %.2f | Rate: %.2f bpp)" % (wspsnr_prop, wsssim_prop, bpp_prop), fontsize=10)
    axes[2].axis('off')

    # Adjust layout and render
    plt.tight_layout()
    plt.show()