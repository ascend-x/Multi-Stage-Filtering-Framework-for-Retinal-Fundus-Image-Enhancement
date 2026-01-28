import cv2
import numpy as np
import os

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
    return image

def save_image(image, path):
    cv2.imwrite(path, image)

def calculate_psnr(original, processed):
    return cv2.PSNR(original, processed)

def calculate_ssim(img1, img2):
    """
    Custom implementation of Structural Similarity Index (SSIM) using NumPy/OpenCV.
    Assuming images are same size and type.
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    # Kernel for local mean
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
               
    return ssim_map.mean()

def calculate_entropy(image):
    """Calculates Shannon Entropy of the image using histogram."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    
    # Filter out zero probabilities to avoid log(0)
    hist = hist[hist > 0]
    
    return -np.sum(hist * np.log2(hist))

def calculate_cii(original, enhanced):
    """
    Calculates Contrast Improvement Index (CII).
    CII = C_enhanced / C_original
    Where Contrast C = (mean of local variance) or similar.
    Here we use global standard deviation as a proxy for global contrast.
    """
    if len(original.shape) == 3:
        orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig = original

    if len(enhanced.shape) == 3:
        enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        enh = enhanced

    c_orig = np.std(orig.astype(float))
    c_enh = np.std(enh.astype(float))
    
    if c_orig == 0: return 0
    return c_enh / c_orig

def generate_synthetic_fundus(width=512, height=512):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        intensity = 255 - int(50 * (y / height))
        img[y, :, 2] = intensity
        img[y, :, 1] = int(intensity * 0.6)
        img[y, :, 0] = int(intensity * 0.1)

    num_vessels = 15
    for _ in range(num_vessels):
        pt1 = (np.random.randint(0, width), np.random.randint(0, height))
        pt2 = (np.random.randint(0, width), np.random.randint(0, height))
        thickness = np.random.randint(1, 4)
        cv2.line(img, pt1, pt2, (10, 40, 100), thickness)

    img = cv2.GaussianBlur(img, (21, 21), 0)
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img
