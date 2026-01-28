import cv2
import numpy as np

class ImageEnhancementPipeline:
    def __init__(self):
        """
        Initializes the optimized classical pipeline.
        Focuses on Luminance (L) channel processing to preserve color fidelity.
        """
        self.clahe_clip = 2.0
        self.clahe_grid = (8, 8)
        self.bilateral_d = 5
        self.bilateral_sigma_color = 50
        self.bilateral_sigma_space = 50
        self.sharpen_amount = 0.8  # Milder sharpening

    def run(self, image):
        """
        Runs the optimized pipeline:
        1. Convert to LAB.
        2. Apply CLAHE to L-channel (Contrast & Illumination).
        3. Denoise L-channel.
        4. Sharpen (optional, light).
        5. Merge and convert back.
        """
        # 1. Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 2. Contrast & Illumination Correction (CLAHE)
        # CLAHE is excellent for handling uneven illumination while improving local contrast
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        l_enhanced = clahe.apply(l)

        # 3. Denoising (Bilateral Filter on L channel)
        # We perform this AFTER CLAHE to avoid enhancing noise too much, 
        # or BEFORE if noise is very heavy. Here we do it after to smooth out HE artifacts if any.
        l_denoised = cv2.bilateralFilter(l_enhanced, 
                                         self.bilateral_d, 
                                         self.bilateral_sigma_color, 
                                         self.bilateral_sigma_space)

        # 4. Sharpening (Unsharp Masking on L channel)
        gaussian = cv2.GaussianBlur(l_denoised, (0, 0), 3)
        l_sharpened = cv2.addWeighted(l_denoised, 1.0 + self.sharpen_amount, gaussian, -self.sharpen_amount, 0)
        
        # 5. Merge components
        # We leave A and B channels untouched to preserve original color tones
        merged_lab = cv2.merge((l_sharpened, a, b))
        
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # Legacy methods are removed/unified into run() for efficiency and coherence in the LAB space
    # The separate stage methods were causing excessive color shifting.
