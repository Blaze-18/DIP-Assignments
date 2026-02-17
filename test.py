import matplotlib.pyplot as plt
import numpy as np
import cv2


def equalize_hist_scratch(img):
    
    hist = np.zeros(256)
    for i in img.flatten():
        hist[i] += 1
    
    pdf = hist / img.size
    
    cdf = np.cumsum(pdf)
    
    new_values = np.round(cdf * 255).astype(np.uint8)
    equalized_img = new_values[img]
    
    return equalized_img

def equalize_hist_cv2(img):
    equalized_img = cv2.equalizeHist(img)
    return equalized_img
    

def plot_images_with_histogram(img, title, position):
    plt.subplot(2, 3, position)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.subplot(2, 3, position + 3)
    plt.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title("Histogram")
def main():
    img = cv2.imread("images/image2.png", 0)
    
    scratch_eq =  equalize_hist_scratch(img)
    cv2_eq = equalize_hist_cv2(img)
    plt.figure(figsize=(12,6))
    plot_images_with_histogram(img, "original", 1)
    plot_images_with_histogram(scratch_eq, "Scratch Eq", 2)
    plot_images_with_histogram(cv2_eq, "CV2 Eq", 3)
    
    plt.tight_layout()
    plt.savefig("output_images/test_equalization.png")
    
        
    
if __name__ == "__main__":
    main()
