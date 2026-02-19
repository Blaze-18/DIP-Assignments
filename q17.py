import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt


# ==========================================================
# 1. LOAD IMAGE
# ==========================================================
def load_image(path):
    image = cv2.imread(path, 0)  # Load as grayscale
    return image


# ==========================================================
# 2. DFT (Discrete Fourier Transform)
# ==========================================================
def apply_dft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))
    return magnitude


# ==========================================================
# 3. DCT (Discrete Cosine Transform)
# ==========================================================
def apply_dct(image):
    image_float = np.float32(image) / 255.0
    dct = cv2.dct(image_float)
    magnitude = np.log(1 + np.abs(dct))
    return magnitude


# ==========================================================
# 4. DWT (Discrete Wavelet Transform)
# ==========================================================
def apply_dwt(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Combine all parts into one image for display
    top = np.hstack((LL, LH))
    bottom = np.hstack((HL, HH))
    combined = np.vstack((top, bottom))

    return combined


# ==========================================================
# 5. DISPLAY FUNCTION
# ==========================================================
def show_result(title, image):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


# ==========================================================
# 6. MAIN FUNCTION
# ==========================================================
def main():

    # Change this to your favorite grayscale image
    image = load_image("image.jpg")

    # Show original image
    show_result("Original Image", image)

    # Apply DFT
    dft_result = apply_dft(image)
    show_result("DFT Magnitude Spectrum", dft_result)

    # Apply DCT
    dct_result = apply_dct(image)
    show_result("DCT Result", dct_result)

    # Apply DWT
    dwt_result = apply_dwt(image)
    show_result("DWT Result (LL, LH, HL, HH)", dwt_result)


if __name__ == "__main__":
    main()
