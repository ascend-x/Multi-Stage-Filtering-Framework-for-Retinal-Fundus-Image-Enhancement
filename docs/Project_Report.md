# Multi-Stage Filtering Framework for Retinal Fundus Image Enhancement (Week 1 Report)

## 1. Abstract
Retinal fundus imaging is a crucial diagnostic tool for ophthalmology. However, raw images often suffer from uneven illumination, noise, and poor contrast. This project implements a **Multi-Stage Filtering Framework** using classical image processing techniques to enhance the quality of these images. The pipeline is designed to be computationally efficient and robust, utilizing modular stages for noise reduction, illumination correction, contrast enhancement, and structure-preserving sharpening.

## 2. Methodology
The framework has been optimized to work in the **LAB Color Space**, processing primarily the Luminance (L) channel to preserve color fidelity.

### Optimized Pipeline Stages
1.  **Color Space Conversion**: The image is converted from BGR to **LAB** (Luminance, A, B).
2.  **Contrast & Illumination Correction (CLAHE)**: 
    -   Applied to the **L-Channel**.
    -   Parameters: Clip Limit 2.0, Grid Size (8,8).
    -   This step effectively handles local contrast enhancement and corrects non-uniform illumination without shifting the image colors.
3.  **Noise Reduction**: 
    -   **Bilateral Filtering** applied to the enhanced L-channel.
    -   Parameters: $d=5, \sigma_{color}=50, \sigma_{space}=50$.
    -   This preserves vessel edges while smoothing out noise and histograms artifacts.
4.  **Sharpening**: 
    -   **Unsharp Masking** on the L-channel.
    -   Weight: 0.8 (Conservative sharpening to avoid halo artifacts).
5.  **Reconstruction**: The processed L-channel is merged with the original A and B channels and converted back to BGR.

## 3. Evaluation Metrics
The enhancement quality is quantitatively evaluated using:
1.  **PSNR (Peak Signal-to-Noise Ratio):** Measures the fidelity of the enhanced image.
2.  **SSIM (Structural Similarity Index):** Assesses the perceptual quality and structural preservation.
3.  **Entropy:** Quantifies the information content. Higher entropy typically indicates better visibility of texture and details.
4.  **CII (Contrast Improvement Index):** The ratio of the contrast of the enhanced image to the original image ($C_{enhanced} / C_{original}$).

## 4. Results & Discussion
The optimized pipeline was tested on sample fundus images.

-   **Visual Analysis:** The enhanced images show significantly improved vessel delineation and uniform background illumination compared to the noisy, possibly vignetted observations. Color tones remain natural (no "washed-out" effect).
-   **Quantitative Analysis (Sample):**
    -   **PSNR:** ~24.22 dB (High fidelity).
    -   **SSIM:** ~0.73 (Good structural preservation).
    -   **CII:** > 1.0 (Contrast improved).
    -   **Entropy:** Increased slightly, indicating more visible details.

## 5. Conclusion
This strictly classical approach demonstrates that significant quality improvements can be achieved without the need for complex machine learning models. The 4-stage pipeline is modular, interpretable, and suitable for deployment in resource-constrained medical imaging environments.

## 6. Future Scope
-   Integration with vessel segmentation algorithms (Week 2).
-   Optimization for real-time video funduscopy.
