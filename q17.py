import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt


# ==========================================================
# 1. LOAD IMAGE
# ==========================================================
def load_image(path):
    image = cv2.imread(path, 0)
    return image


# ==========================================================
# 2. DFT
# ==========================================================
def apply_dft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))
    return magnitude


# ==========================================================
# 3. DCT
# ==========================================================
def apply_dct(image):
    image_float = np.float32(image) / 255.0
    dct = cv2.dct(image_float)
    magnitude = np.log(1 + np.abs(dct))
    return magnitude


# ==========================================================
# 4. DWT
# ==========================================================
def apply_dwt(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH


# ==========================================================
# 5. DISPLAY FUNCTIONS
# ==========================================================

def show_all_transforms(images, titles):

    plt.figure(figsize=(15, 8))
    
    # Simple loop through all images
    for i in range(len(images)):
        plt.subplot(2, 4, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("output_images/DFT_DCT_DWT_transforms")


# ==========================================================
# 6. MAIN FUNCTION
# ==========================================================
def main():

    image = load_image("images/image2.png")
    # Apply DFT
    dft_image = apply_dft(image)

    # Apply DCT
    dct_image = apply_dct(image)

    # Apply DWT
    LL, LH, HL, HH = apply_dwt(image)
    
    images = [image, dft_image, dct_image, LL, LH, HL, HH]
    titles = [
            "Original Image", "DFT Image", "DCT Image", 
            "DWT (LL)", "DWT (LH)", "DWT (HL)", "DWT (HH)"
        ]
    
    show_all_transforms(images, titles)


if __name__ == "__main__":
    main()
