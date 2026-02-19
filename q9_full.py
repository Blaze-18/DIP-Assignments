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
# Histogram Matching FROM SCRATCH (Equalization + Inverse method)
# ------------------------------------------------------------
def histogram_matching_scratch_inverse(source, reference):
    """
    Second conceptual implementation of histogram matching.

    Steps:
    1. Equalize the source image using its CDF  → T(r)
    2. Compute CDF transform of reference       → G(z)
    3. Invert reference transform               → G⁻¹(s)
    4. Apply inverse transform to equalized src → final matched image
    """

    # ---------- Step 1: Equalize SOURCE ----------
    src_hist = np.bincount(source.ravel(), minlength=256)
    src_pdf = src_hist / source.size
    src_cdf = np.cumsum(src_pdf)

    # T(r) mapping → equalized image in range [0,255]
    T = np.round(src_cdf * 255).astype(np.uint8)
    equalized_source = T[source]


    # ---------- Step 2: Reference CDF transform ----------
    ref_hist = np.bincount(reference.ravel(), minlength=256)
    ref_pdf = ref_hist / reference.size
    ref_cdf = np.cumsum(ref_pdf)

    # G(z) mapping
    G = np.round(ref_cdf * 255).astype(np.uint8)


    # ---------- Step 3: Invert reference transform ----------
    G_inv = np.zeros(256, dtype=np.uint8)

    for s in range(256):
        # find gray level z whose G(z) is closest to s
        diff = np.abs(G - s)
        G_inv[s] = np.argmin(diff)


    # ---------- Step 4: Apply inverse transform ----------
    matched = G_inv[equalized_source]

    return matched


# ------------------------------------------------------------
# Built-in Histogram Matching
# ------------------------------------------------------------
def histogram_matching_builtin(source, reference):
    matched = match_histograms(source, reference)
    return matched.astype(np.uint8)


# ------------------------------------------------------------
# Create LOW, NORMAL, HIGH contrast versions
# ------------------------------------------------------------
def create_contrast_versions(img):
    low = cv2.normalize(img, None, 80, 170, cv2.NORM_MINMAX)
    normal = img.copy()
    high = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return low, normal, high


# ------------------------------------------------------------
# FUNCTION 1: Save ONLY images
# ------------------------------------------------------------
def show_images(images, titles):
    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        plt.subplot(4, 3, i + 1)
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
        plt.subplot(4, 3, i + 1)
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
    src_low, src_norm, src_high = create_contrast_versions(source)

    # Manual matching scratch 1
    manual_1_low = histogram_matching_scratch(src_low, reference)
    manual_1_norm = histogram_matching_scratch(src_norm, reference)
    manual_1_high = histogram_matching_scratch(src_high, reference)
    
    # Manual matching sractch 2
    manual_2_low = histogram_matching_scratch_inverse(source, reference)
    manual_2_norm = histogram_matching_scratch_inverse(source, reference)
    manual_2_high = histogram_matching_scratch_inverse(source, reference)
    
    # Built-in matching
    builtin_low = histogram_matching_builtin(src_low, reference)
    builtin_norm = histogram_matching_builtin(src_norm, reference)
    builtin_high = histogram_matching_builtin(src_high, reference)

    # Include REFERENCE image for comparison
    images = [
        reference, source,
        manual_1_low, manual_1_norm, manual_1_high,
        manual_2_low, manual_2_norm, manual_2_high,
        builtin_low, builtin_norm, builtin_high
    ]

    titles = [
        "Reference", "Source",
        "Manual 1 Low", "Manual 1 Normal", "Manual 1 High",
        "Manual 2 Low", "Manual 2 Normal", "Manual 2 High",
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
