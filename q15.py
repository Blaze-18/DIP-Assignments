import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# 1. IMAGE LOADING
# ==========================================================
def load_image(path):
    image = cv2.imread(path, 0)  # Load as grayscale
    return image


# ==========================================================
# 2. CONTRAST ADJUSTMENT
# ==========================================================
def change_contrast(image, alpha):
    # alpha < 1  → low contrast
    # alpha = 1  → normal contrast
    # alpha > 1  → high contrast
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return new_image


# ==========================================================
# 3. FOURIER TRANSFORM
# ==========================================================
def apply_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift


def apply_ifft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


# ==========================================================
# 4. IDEAL FILTER
# ==========================================================
def ideal_filter(shape, D0, filter_type="low", D1=0):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)

            if filter_type == "low":
                if D <= D0:
                    mask[i, j] = 1

            elif filter_type == "high":
                if D > D0:
                    mask[i, j] = 1

            elif filter_type == "band":
                if D0 < D < D1:
                    mask[i, j] = 1

    return mask


# ==========================================================
# 5. BUTTERWORTH FILTER
# ==========================================================
def butterworth_filter(shape, D0, n, filter_type="low", D1=0):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)

            if filter_type == "low":
                mask[i, j] = 1 / (1 + (D / D0) ** (2 * n))

            elif filter_type == "high":
                mask[i, j] = 1 - (1 / (1 + (D / D0) ** (2 * n)))

            elif filter_type == "band":
                low = 1 / (1 + (D / D0) ** (2 * n))
                high = 1 - (1 / (1 + (D / D1) ** (2 * n)))
                mask[i, j] = low * high

    return mask


# ==========================================================
# 6. GAUSSIAN FILTER
# ==========================================================
def gaussian_filter(shape, D0, filter_type="low", D1=0):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)

            if filter_type == "low":
                mask[i, j] = np.exp(-(D ** 2) / (2 * (D0 ** 2)))

            elif filter_type == "high":
                mask[i, j] = 1 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))

            elif filter_type == "band":
                low = np.exp(-(D ** 2) / (2 * (D0 ** 2)))
                high = 1 - np.exp(-(D ** 2) / (2 * (D1 ** 2)))
                mask[i, j] = low * high

    return mask


# ==========================================================
# 7. APPLY FILTER
# ==========================================================
def apply_filter(image, mask):
    fshift = apply_fft(image)
    filtered = fshift * mask
    result = apply_ifft(filtered)
    return result


# ==========================================================
# 8. DISPLAY FUNCTION
# ==========================================================
def show_images(title, original, filtered):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(filtered, cmap="gray")
    plt.axis("off")

    plt.savefig(f"output_images/{title}")


# ==========================================================
# 9. MAIN FUNCTION
# ==========================================================
def main():
    image = load_image("images/image2.png")

    # Create contrast versions
    low_contrast = change_contrast(image, 0.5)
    normal_contrast = change_contrast(image, 1.0)
    high_contrast = change_contrast(image, 1.5)

    # Choose parameters
    D0 = 30
    D1 = 60
    n_values = [1, 2, 5]

    # Example: Butterworth Low-pass with different n
    for n in n_values:
        mask = butterworth_filter(image.shape, D0, n, "low")
        result = apply_filter(normal_contrast, mask)
        show_images(f"Butterworth LPF (n={n})", normal_contrast, result)

    # Gaussian Low-pass
    mask = gaussian_filter(image.shape, D0, "low")
    result = apply_filter(normal_contrast, mask)
    show_images("Gaussian LPF", normal_contrast, result)

    # Ideal Low-pass
    mask = ideal_filter(image.shape, D0, "low")
    result = apply_filter(normal_contrast, mask)
    show_images("Ideal LPF", normal_contrast, result)

    #  similarly test:
    # "high"
    # "band"


if __name__ == "__main__":
    main()
