import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. STRUCTURING ELEMENTS (5×5)
# ============================================================
def get_structuring_elements():

    rect = np.ones((5, 5), dtype=np.uint8)

    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    # Diamond (manual)
    diamond = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.uint8)

    return {
        "Rectangle": rect,
        "Ellipse": ellipse,
        "Cross": cross,
        "Diamond": diamond
    }


# ============================================================
# 2. SCRATCH EROSION & DILATION
# ============================================================
def erosion_scratch(image, kernel):

    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]

            # erosion → all kernel ones must fit inside foreground
            if np.all(region[kernel == 1] == 255):
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output


def dilation_scratch(image, kernel):

    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]

            # dilation → any kernel one overlaps foreground
            if np.any(region[kernel == 1] == 255):
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output


# ============================================================
# 3. OTHER MORPHOLOGICAL OPS (SCRATCH)
# ============================================================
def opening_scratch(image, kernel):
    return dilation_scratch(erosion_scratch(image, kernel), kernel)


def closing_scratch(image, kernel):
    return erosion_scratch(dilation_scratch(image, kernel), kernel)


def tophat_scratch(image, kernel):
    return image - opening_scratch(image, kernel)


def blackhat_scratch(image, kernel):
    return closing_scratch(image, kernel) - image


# ============================================================
# 4. BUILT-IN MORPHOLOGY
# ============================================================
def morphology_builtin(image, kernel):

    results = {}

    results["Erosion"] = cv2.erode(image, kernel)
    results["Dilation"] = cv2.dilate(image, kernel)
    results["Opening"] = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    results["Closing"] = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    results["Top Hat"] = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    results["Black Hat"] = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    return results


# ============================================================
# 5. SCRATCH MORPHOLOGY WRAPPER
# ============================================================
def morphology_scratch(image, kernel):

    results = {}

    results["Erosion"] = erosion_scratch(image, kernel)
    results["Dilation"] = dilation_scratch(image, kernel)
    results["Opening"] = opening_scratch(image, kernel)
    results["Closing"] = closing_scratch(image, kernel)
    results["Top Hat"] = tophat_scratch(image, kernel)
    results["Black Hat"] = blackhat_scratch(image, kernel)

    return results


# ============================================================
# 6. DISPLAY FUNCTION
# ============================================================
def show_results(title, original, results_dict):

    os.makedirs("output_images", exist_ok=True)

    total = len(results_dict) + 1
    cols = 4
    rows = int(np.ceil(total / cols))

    plt.figure(figsize=(12, 3 * rows))

    # original
    plt.subplot(rows, cols, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # processed
    for i, (name, img) in enumerate(results_dict.items(), start=2):
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"output_images/{title}.png")
    plt.close()

    print(f"{title} saved.")


# ============================================================
# 7. MAIN
# ============================================================
def main():

    image = cv2.imread("images/image2.png", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return

    # convert to binary (important for morphology)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    elements = get_structuring_elements()

    for name, kernel in elements.items():

        # Built-in
        builtin_results = morphology_builtin(binary, kernel)
        show_results(f"builtin_{name}", binary, builtin_results)

        # Scratch
        scratch_results = morphology_scratch(binary, kernel)
        show_results(f"scratch_{name}", binary, scratch_results)

    print("\nMorphological transformations completed!")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
