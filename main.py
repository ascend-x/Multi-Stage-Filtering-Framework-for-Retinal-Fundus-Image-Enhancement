import argparse
import cv2
import os
import sys
import matplotlib.pyplot as plt
from src.utils import load_image, save_image, calculate_psnr, calculate_ssim, calculate_entropy, calculate_cii, generate_synthetic_fundus
from src.pipeline import ImageEnhancementPipeline

def main():
    parser = argparse.ArgumentParser(description="Multi-Stage Filtering Framework for Retinal Fundus Image Enhancement (Classical)")
    parser.add_argument('--input', type=str, help='Path to input image. If not provided, a synthetic image is generated.')
    parser.add_argument('--output', type=str, default='output/result.png', help='Path to save the output image.')
    parser.add_argument('--plot', action='store_true', help='Plot the comparison result.')
    
    args = parser.parse_args()
    
    # 1. Load or Generate Image
    if args.input:
        print(f"Loading image from {args.input}...")
        try:
            image = load_image(args.input)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("No input provided. Generating synthetic retinal fundus image...")
        image = generate_synthetic_fundus()
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/synthetic_input.png", image)
        print("Saved synthetic input to output/synthetic_input.png")

    # 2. Run Pipeline
    print("Running multi-stage image processing pipeline...")
    pipeline = ImageEnhancementPipeline()
    enhanced_image = pipeline.run(image)

    # 3. Evaluation
    print("Computing performance metrics...")
    psnr = calculate_psnr(image, enhanced_image)
    ssim_val = calculate_ssim(image, enhanced_image)
    
    orig_entropy = calculate_entropy(image)
    new_entropy = calculate_entropy(enhanced_image)
    
    cii = calculate_cii(image, enhanced_image)

    print("-" * 40)
    print(f" Performance Metrics")
    print("-" * 40)
    print(f" PSNR (dB)       : {psnr:.2f}")
    print(f" SSIM            : {ssim_val:.4f}")
    print(f" Entropy (Orig)  : {orig_entropy:.4f}")
    print(f" Entropy (Enh)   : {new_entropy:.4f}")
    print(f" CII (Contrast)  : {cii:.4f}")
    print("-" * 40)

    # 4. Save Result
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    save_image(enhanced_image, args.output)
    print(f"Enhanced image saved to {args.output}")

    # 5. Visualization
    # Create a comparison plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert BGR to RGB for Matplotlib
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enh_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    
    ax[0].imshow(img_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(enh_rgb)
    ax[1].set_title("Enhanced Image\n(PSNR: {:.2f} dB, SSIM: {:.2f})".format(psnr, ssim_val))
    ax[1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")
    
    if args.plot:
        plt.show()

if __name__ == "__main__":
    main()
