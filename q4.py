import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Manual Binary Threshold
# ------------------------------------------------------------
def binary_threshold(img, T=127):
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            if img[i, j] >= T:
                result[i, j] = 255
            else:
                result[i, j] = 0
    
    return result


# ------------------------------------------------------------
# Manual Binary Inverse Threshold
# ------------------------------------------------------------
def binary_inverse_threshold(img, T=127):
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            if img[i, j] >= T:
                result[i, j] = 0
            else:
                result[i, j] = 255
    
    return result


# ------------------------------------------------------------
# Manual Truncate Threshold
# ------------------------------------------------------------
def truncate_threshold(img, T=127):
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            if img[i, j] > T:
                result[i, j] = T
            else:
                result[i, j] = img[i, j]
    
    return result

def adaptive_mean_threshold(img, block_size=11, C=2):
    """
    Threshold value is the mean of the neighborhood minus C.
    """
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)
    pad = block_size // 2
    # Padding the image to handle pixels at the edges
    padded_img = np.pad(img, pad, mode='reflect')
    
    for i in range(h):
        for j in range(w):
            # Extract neighborhood window
            roi = padded_img[i:i + block_size, j:j + block_size]
            # Calculate local threshold: mean - C
            threshold = np.mean(roi) - C
            # Apply binary thresholding
            if img[i, j] > threshold:
                result[i,j] = 255
            else:
                result[i,j] = 0
            
    return result

# ------------------------------------------------------------
# Function to apply all manual thresholds
# ------------------------------------------------------------
def apply_thresholds(img):
    # 1. Manual Binary Threshold
    thresh_binary = binary_threshold(img, 127)

    # 2. Manual Binary Inverse Threshold
    thresh_inverse = binary_inverse_threshold(img, 127)

    # 3. Manual Truncate Threshold
    thresh_trunc = truncate_threshold(img, 127)
    
    # 4. Manual Adaptive mean Threshold
    thresh_adaptive_mean =  adaptive_mean_threshold(img, block_size=11, C=2)

    return thresh_binary, thresh_inverse, thresh_trunc, thresh_adaptive_mean


# ------------------------------------------------------------
# Log Transformation
# ------------------------------------------------------------
def log_transform(img):
    img_float = img / 255.0
    # 2. Apply log: s = log(1 + r)
    log_img = np.log(1 + img_float)
    # 3. Scale the log(2) result back to 1.0
    # This is the "c" constant equivalent
    log_img_normalized = log_img / np.log(2)
    # 4. Convert back to 0-255 uint8
    return np.uint8(log_img_normalized * 255)


# ------------------------------------------------------------
# Power (Gamma) Transformation
# ------------------------------------------------------------
def gamma_transform(img, gamma=0.5):
    img_float = img / 255.0
    gamma_img = np.power(img_float, gamma)

    gamma_img = np.uint8(gamma_img * 255)

    return gamma_img


# ------------------------------------------------------------
# FUNCTION 1: Show only images (with for loop)
# ------------------------------------------------------------
def show_images(images, titles):
    plt.figure(figsize=(15, 8))
    
    # Use for loop to display all images
    for i in range(len(images)):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("output_images/images_only.png")
    plt.close()
    print("Images saved to output_images/images_only.png")


# ------------------------------------------------------------
# FUNCTION 2: Show only histograms (with for loop)
# ------------------------------------------------------------
def show_histograms(images, titles):
    plt.figure(figsize=(15, 8))
    
    # Use for loop to display all histograms
    for i in range(len(images)):
        plt.subplot(3, 3, i+1)
        plt.hist(images[i].ravel(), bins=256, range=[0, 256], color='gray')
        plt.title(f"{titles[i]} Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("output_images/histograms_only.png")
    plt.close()
    print("Histograms saved to output_images/histograms_only.png")


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    # Step 1: Read grayscale image
    img = cv2.imread("images/image2.png", cv2.IMREAD_GRAYSCALE)

    # Step 2: Apply manual thresholding
    thresh_binary, thresh_inverse, thresh_trunc, adaptive_mean = apply_thresholds(img)

    # Step 3: Apply transformations
    log_img = log_transform(img)
    gamma_img = gamma_transform(img, gamma=0.5)

    # Step 4: Create a single list with all converted images
    # Order: Original + 3 thresholded + 2 transformed = 6 images
    converted_images = [img, thresh_binary, thresh_inverse,    
    thresh_trunc, adaptive_mean, log_img, gamma_img]
    
    # Step 5: Create titles for the images
    image_titles = ["Original", "Binary", "Inverse", "Truncate", "AdaptiveMean","Log", "Gamma"]

    # Step 6: Call the functions with the single list
    show_images(converted_images, image_titles)
    show_histograms(converted_images, image_titles)
    
    print("All outputs saved successfully!")


# ------------------------------------------------------------
# Run the program
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
