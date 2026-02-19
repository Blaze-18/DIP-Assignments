import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. OWN FILTER2D IMPLEMENTATION (from scratch)
# ============================================================
def filter2d_scratch(image, kernel):
    """
    Simple manual implementation of cv2.filter2D()
    Uses zero padding and convolution.
    """

    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    """ mode="constant" for zero padding 
        (In all cases use reflect as this is  the  safest)
    """
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    output = np.zeros_like(image, dtype=np.float32)

    # Convolution
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)

    # Clip values to valid range
    output = np.clip(output, 0, 255)

    return output.astype(np.uint8)


# ============================================================
# 2. KERNEL DEFINITIONS
# ============================================================

def get_kernels():
    # Average kernel
    avg = np.ones((3, 3)) / 9

    # Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Prewitt
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

    # Scharr
    scharr_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]])

    scharr_y = np.array([[-3, -10, -3],
                         [0, 0, 0],
                         [3, 10, 3]])

    # Laplace
    laplace = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

    # -------- 4 Custom Kernels --------
    sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    emboss = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])

    edge_enhance = np.array([[0, 0, 0],
                             [-1, 1, 0],
                             [0, 0, 0]])

    box_blur_5 = np.ones((5, 5)) / 25

    return {
        "Average": avg,
        "Sobel X": sobel_x,
        "Sobel Y": sobel_y,
        "Prewitt X": prewitt_x,
        "Prewitt Y": prewitt_y,
        "Scharr X": scharr_x,
        "Scharr Y": scharr_y,
        "Laplace": laplace,
        "Sharpen (custom)": sharpen,
        "Emboss (custom)": emboss,
        "Edge Enhance (custom)": edge_enhance,
        "Box Blur 5x5 (custom)": box_blur_5,
    }


# ============================================================
# 3. APPLY OPENCV FILTERS
# ============================================================
def apply_opencv_filters(image, kernels):
    results = {}

    for name, kernel in kernels.items():

        # Skip custom kernels for built-in requirement
        if "custom" in name.lower() or "Box Blur" in name:
            continue

        filtered = cv2.filter2D(image, -1, kernel)
        results[name] = filtered

    return results


# ============================================================
# 4. APPLY SCRATCH FILTERS
# ============================================================
def apply_scratch_filters(image, kernels):
    results = {}

    for name, kernel in kernels.items():
        filtered = filter2d_scratch(image, kernel)
        results[name] = filtered

    return results


# ============================================================
# 5. DISPLAY & SAVE RESULTS
# ============================================================
def show_results(title, images_dict, original):
    os.makedirs("output_images", exist_ok=True)

    total = len(images_dict) + 1
    cols = 4
    rows = int(np.ceil(total / cols))

    plt.figure(figsize=(12, 3 * rows))

    # Original image
    plt.subplot(rows, cols, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Filtered images
    for i, (name, img) in enumerate(images_dict.items(), start=2):
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"output_images/{title}.png")
    plt.close()

    print(f"{title} saved.")


# ============================================================
# 6. MAIN FUNCTION
# ============================================================
def main():

    image = cv2.imread("images/image2.png", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return

    kernels = get_kernels()

    # ----- Part 1: OpenCV filtering -----
    opencv_results = apply_opencv_filters(image, kernels)
    show_results("opencv_filters", opencv_results, image)

    # ----- Part 2: Scratch filtering -----
    scratch_results = apply_scratch_filters(image, kernels)
    show_results("scratch_filters", scratch_results, image)

    print("\nSpatial filtering experiment completed!")


# ============================================================
# RUN PROGRAM
# ============================================================
if __name__ == "__main__":
    main()
