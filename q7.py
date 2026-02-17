import cv2
import numpy as np
import matplotlib.pyplot as plt



# ------------------------------------------------------------
# Histogram Equalization FROM SCRATCH
# ------------------------------------------------------------
def histogram_equalization_scratch(img):
    # Step 1: Calculate histogram (256 intensity values)
    hist = np.zeros(256)
    # count of each pixel
    for pixel in img.flatten():
        hist[pixel] += 1

    # Step 2: Calculate probability (normalize histogram) (PDF)
    hist = hist / img.size

    # Step 3: Calculate cumulative distribution function (CDF)
    cdf = np.cumsum(hist)

    # Step 4: Create new pixel mapping using CDF
    new_values = np.round(cdf * 255).astype(np.uint8)

    # Step 5: Replace old pixels with new equalized pixels
    equalized_img = new_values[img]

    return equalized_img


# ------------------------------------------------------------
# OpenCV Histogram Equalization
# ------------------------------------------------------------
def histogram_equalization_opencv(img):
    return cv2.equalizeHist(img)


# ------------------------------------------------------------
# Function to show image and histogram
# ------------------------------------------------------------
def show_image_and_histogram(title, img, position):
    plt.subplot(2, 3, position)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.subplot(2, 3, position + 3)
    plt.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title("Histogram")


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    # CHANGE this path to your image file
    image_path = "images/image2.png"

    # Step 1: Read image
    img = cv2.imread(image_path, 0)

    # Step 2: Apply histogram equalization
    scratch_eq = histogram_equalization_scratch(img)
    opencv_eq = histogram_equalization_opencv(img)

    # Step 3: Display comparison
    plt.figure(figsize=(12, 6))

    show_image_and_histogram("Original Image", img, 1)
    show_image_and_histogram("Scratch Equalization", scratch_eq, 2)
    show_image_and_histogram("OpenCV Equalization", opencv_eq, 3)

    plt.tight_layout()
    plt.savefig("output_images/histogram_equalization.png")


# ------------------------------------------------------------
# Run the program
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

