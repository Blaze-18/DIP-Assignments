import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_dft(img):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift

def apply_idft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)
    idft = np.fft.ifft2(dft_ishift)
    img = np.abs(idft)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
    
def apply_filter(img, kernel):
    dft_shift = apply_dft(img)
    filtered = dft_shift * kernel
    return apply_idft(filtered)
    
def ideal_kernel(shape, filter_type, d0, d1):
    row, col = shape
    crow, ccol = row // 2, col // 2
    
    kernel = np.zeros((row, col))
    
    for i in range(row):
        for j in range(col):
            D = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if filter_type == "low":
                if D < d0:
                    kernel[i,j] = 1
            elif filter_type == "high":
                if D > d0:
                    kernel[i,j] = 1
            elif filter_type == "band":
                if d0 < D < d1:
                    kernel[i,j] = 1
    return kernel
    
def gaussian_kernel(shape, filter_type, d0, d1):
    row, col = shape
    crow, ccol = row // 2, col // 2
    
    kernel = np.zeros((row, col))
    
    for i in range(row):
        for j in range(col):
            D = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if D == 0:
                if filter_type == "low" :
                    kernel[i,j] = 1
                elif filter_type == "high" or filter_type == "band":
                    kernel[i,j] = 0
                continue
                
            if filter_type == "low":
                # formula
                kernel[i,j] = np.exp(- ( (D**2)/(2 * (d0 ** 2) ) ) )
            elif filter_type == "high":
                    kernel[i,j] = 1 - np.exp(- ( (D**2)/(2 * (d0 ** 2) ) ) )
            elif filter_type == "band":
                low = np.exp(- ( (D**2)/(2 * (d0 ** 2) ) ) )
                high = 1 - np.exp(- ( (D**2)/(2 * (d1 ** 2) ) ) )
                kernel[i,j] = low * high
    return kernel

def buttersworth_kernel(shape, filter_type, d0, d1, n=2):
    row, col = shape
    crow, ccol = row // 2, col // 2
    
    kernel = np.zeros((row, col))
    
    for i in range(row):
        for j in range(col):
            D = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if D == 0:
                if filter_type == "low" :
                    kernel[i,j] = 1
                elif filter_type == "high" or filter_type == "band":
                    kernel[i,j] = 0
                continue
            if filter_type == "low":
                kernel[i,j] = 1 / (1 + (D / d0)**(2*n))
            elif filter_type == "high":
                    kernel[i,j] = 1 - 1 / (1 + (D / d0)**(2*n))
            elif filter_type == "band":
                low = 1 / (1 + (D / d0)**(2*n))
                high = 1 - 1 / (1 + (D / d1)**(2*n))
                kernel[i,j] = low * high
    return kernel
    
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

    plt.tight_layout()
    plt.savefig(f"fd_output/{title}.png")
    plt.close()

def main():
    image = cv2.imread("images/fracture.png", 0)  # Load as grayscale

    D0 = 30
    D1 = 60
    n_values = [1, 2, 5]

    # Butterworth Low-pass with different n
    for n in n_values:
        kernel = buttersworth_kernel(image.shape, "low", D0, D1, n)
        result = apply_filter(image, kernel)
        show_images(f"Butterworth LPF (n={n})", image, result)

    # Gaussian Low-pass
    kernel = gaussian_kernel(image.shape, "low", D0, D1)
    result = apply_filter(image, kernel)
    show_images("Gaussian LPF", image, result)

    # Ideal Low-pass
    kernel = ideal_kernel(image.shape, "low", D0, D1)
    result = apply_filter(image, kernel)
    show_images("Ideal LPF", image, result)

    # High-pass filters
    kernel = buttersworth_kernel(image.shape, "high", D0, D1, n=2)
    result = apply_filter(image, kernel)
    show_images("Butterworth HPF (n=2)", image, result)

    kernel = gaussian_kernel(image.shape, "high", D0, D1)
    result = apply_filter(image, kernel)
    show_images("Gaussian HPF", image, result)

    kernel = ideal_kernel(image.shape, "high", D0, D1)
    result = apply_filter(image, kernel)
    show_images("Ideal HPF", image, result)

    # Band-pass filters
    kernel = buttersworth_kernel(image.shape, "band", D0, D1, n=2)
    result = apply_filter(image, kernel)
    show_images("Butterworth BPF (n=2)", image, result)

    kernel = gaussian_kernel(image.shape, "band", D0, D1)
    result = apply_filter(image, kernel)
    show_images("Gaussian BPF", image, result)

    kernel = ideal_kernel(image.shape, "band", D0, D1)
    result = apply_filter(image, kernel)
    show_images("Ideal BPF", image, result)


if __name__ == "__main__":
    main()
