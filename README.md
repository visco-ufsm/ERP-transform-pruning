# Geometry-Aware Transform Pruning for Omnidirectional Image Compression

This repository contains the reference implementation associated with the paper:

**“Geometry-Aware Transform Pruning for Omnidirectional Image Compression”**  submitted to *IEEE Access*.

## Authors

- Thiago L. T. da Silveira (UFSM, Brazil)
- Enzo B. Segala (UFRGS, Brazil)
- Fábio M. Bayer (UFSM, Brazil)
- R. J. Cintra (UFPE, Brazil)

## Overview

This repository provides a reference implementation of a geometry-aware transform pruning framework for efficient compression of omnidirectional (360°) images represented in the Equirectangular Projection (ERP) format.

The proposed method exploits the latitude-dependent sampling characteristics of ERP images to adaptively reduce the number of retained transform basis functions while maintaining compatibility with a JPEG-like transform coding pipeline. The approach reduces transform complexity while preserving perceptual quality, evaluated using spherical quality metrics.

The implementation includes:

- A baseline JPEG-like 8×8 DCT transform codec;
- The proposed latitude-adaptive transform pruning codec;
- JPEG-style quantization and dequantization;
- Weighted Spherical PSNR (WS-PSNR) and Weighted Spherical SSIM (WS-SSIM) evaluation;
- Estimated bitrate computation.

## Repository Structure

```text
.
├── main.py              # Experimental script and result visualization
├── codec.py             # Proposed method and JPEG baseline implementations
├── metrics.py           # Rate and spherical quality metrics
├── utils.py             # Image block partitioning/reconstruction utilities
└── requirements.txt     # Python dependencies
```

## Installation

The code requires Python 3.x and the dependencies listed in `requirements.txt`.

Install the required packages using:

```bash
pip install -r requirements.txt
```
## Running the Experiment

The example implementation can be executed using:

```bash
python main.py
```

## Notes

The reported execution times correspond to high-level NumPy operations and should not be interpreted as optimized codec runtimes.

The implementation is intended as a reference implementation for reproducing the proposed geometry-aware transform pruning method and evaluating its computational behavior. The code prioritizes clarity and reproducibility over low-level optimization.

The reported bitrate (bpp) is a simplified estimate based on the number of non-zero quantized transform coefficients. It is used as a proxy for compression efficiency and does not represent the final bitrate of a complete JPEG bitstream with entropy coding.

## Citation

If you use this code in your research, please cite the associated paper:

```bibtex
@article{silveira2026geometry,
  title={Geometry-Aware Transform Pruning for Omnidirectional Image Compression},
  author={Silveira, Thiago L. T. da and Segala, Enzo B. and Bayer, F{\'a}bio M. and Cintra, R. J.},
  journal={IEEE Access},
  note = {Submitted},
  year={2026}
}
```
