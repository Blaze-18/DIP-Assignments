import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms


# ------------------------------------------------------------
# Manual Histogram Matching (CDF based)
# ------------------------------------------------------------
def histogram_matching_scratch(source, reference):

    src_hist = np.zeros(256)
    for pixel in source.flatten():
        src_hist[pixel] += 1

    src_pdf = src_hist / source.size
    src_cdf = np.cumsum(src_pdf)

    ref_hist = np.zeros(256)
    for pixel in reference.flatten():
        ref_hist[pixel] += 1

    ref_pdf = ref_hist / reference.size
    ref_cdf = np.cumsum(ref_pdf)

    mapping = np.zeros(256, dtype=np.uint8)

    # Beginner-friendly matching loop
    for i in range(256):
        min_diff = 1.0
        best_match = 0

        for j in range(256):
            diff = abs(src_cdf[i] - ref_cdf[j])
            if diff < min_diff:
                min_diff = diff
                best_match = j

        mapping[i] = best_match

    return mapping[source]


# ------------------------------------------------------------
# Built-in Histogram Matching
# ------------------------------------------------------------
def histogram_matching_builtin(source, reference):
    matched = match_histograms(source, reference)
    return matched.astype(np.uint8)


# ------------------------------------------------------------
# Create LOW, NORMAL, HIGH contrast versions
# ------------------------------------------------------------
def change_contrast(image, alpha):
    # alpha < 1  → low contrast (0.5)
    # alpha = 1  → normal contrast (1.0)
    # alpha > 1  → high contrast (1.5)
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return new_image


# ------------------------------------------------------------
# FUNCTION 1: Save ONLY images
# ------------------------------------------------------------
def show_images(images, titles):
    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("output_images/matching_images.png")
    plt.close()
    print("Images saved to output_images/matching_images.png")


# ------------------------------------------------------------
# FUNCTION 2: Save ONLY histograms
# ------------------------------------------------------------
def show_histograms(images, titles):
    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        plt.subplot(3, 3, i + 1)
        plt.hist(images[i].ravel(), bins=256, range=[0, 256], color="gray")
        plt.title(f"{titles[i]} Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("output_images/matching_histograms.png")
    plt.close()
    print("Histograms saved to output_images/matching_histograms.png")


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():

    source = cv2.imread("images/source.png", cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread("images/reference.png", cv2.IMREAD_GRAYSCALE)

    if source is None or reference is None:
        print("Error: Check image paths.")
        return


    # Contrast versions of source
    src_low = change_contrast(source, 0.5)
    src_norm = change_contrast(source, 1.0)
    src_high = change_contrast(source, 1.5)

    # Manual matching
    manual_low = histogram_matching_scratch(src_low, reference)
    manual_norm = histogram_matching_scratch(src_norm, reference)
    manual_high = histogram_matching_scratch(src_high, reference)

    # Built-in matching
    builtin_low = histogram_matching_builtin(src_low, reference)
    builtin_norm = histogram_matching_builtin(src_norm, reference)
    builtin_high = histogram_matching_builtin(src_high, reference)

    # Include REFERENCE image for comparison
    images = [
        reference, source,
        manual_low, manual_norm, manual_high,
        builtin_low, builtin_norm, builtin_high
    ]

    titles = [
        "Reference", "Source",
        "Manual Low", "Manual Normal", "Manual High",
        "Builtin Low", "Builtin Normal", "Builtin High"
    ]

    show_images(images, titles)
    show_histograms(images, titles)

    print("Histogram matching visualization completed!")


# ------------------------------------------------------------
# Run program
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

