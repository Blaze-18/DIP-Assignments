# ============================================
# Canny Edge Detection (Grayscale Image)
# ============================================

import cv2
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Function to apply Canny Edge Detection
# ------------------------------------------------------------
def apply_canny(image_path, threshold1=100, threshold2=200):
    
    # Read image directly in grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if gray is None:
        print("Error: Image not found.")
        return
    
    # Optional: Apply Gaussian Blur (recommended)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # -----------------------------------------
    # Visualization
    # -----------------------------------------
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f"Canny Edges\nT1={threshold1}, T2={threshold2}")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("output_images/canny_edge.png")


# ------------------------------------------------------------
# Main Program
# ------------------------------------------------------------
if __name__ == "__main__":
    
    image_path = "input.jpg"   # Change to your image name
    
    # Try different thresholds
    apply_canny(image_path, threshold1=50, threshold2=150)
