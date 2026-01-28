# Multi-Stage Filtering Framework for Retinal Fundus Image Enhancement (Week 1 Scope)

## Overview
This project presents a classical image processing pipeline developed to enhance retinal fundus images. It employs a multi-stage approach including noise reduction, illumination correction, contrast enhancement, and sharpening to improve the visibility of retinal structures. 

**Note:** This implementation uses only standard libraries (OpenCV, NumPy, Matplotlib) and contains **no Machine Learning components**, focusing on signal processing fundamentals.

## Methodology (Optimized LAB Pipeline)
1.  **Color Space Conversion**: Convert input to LAB to isolate Luminance.
2.  **Contrast & Illumination**: Apply CLAHE to the L-channel.
3.  **Noise Reduction**: Apply Bilateral filtering to the L-channel.
4.  **Sharpening**: Apply Unsharp Masking to the L-channel.
5.  **Reconstruction**: Merge processed L with original A/B channels.

## Evaluation Metrics
- **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction quality.
- **SSIM (Structural Similarity Index):** Measures structural retention.
- **Entropy:** Measures information content / texture detail.
- **CII (Contrast Improvement Index):** Ratio of contrast in the enhanced image to the original.

## Setup & Execution
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the pipeline:
    ```bash
    python main.py
    ```
    - By default, it generates and processes a synthetic retinal image.
    - To use a real image: `python main.py --input path/to/image.jpg`

## Results
Results including the "Original vs Enhanced" comparison plot are saved in the `output/` directory.
