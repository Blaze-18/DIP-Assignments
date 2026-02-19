# ============================================
# Bit Plane Slicing Program (Beginner Friendly)
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------
# Function to get bit planes
# ---------------------------------
def get_bit_planes(image):
    bit_planes = []
    
    # Loop through 8 bits (0 to 7)
    for i in range(8):
        
        # Extract bit using bitwise AND
        plane = (image >> i) & 1
        
        # Multiply by 255 to make it visible
        plane = plane * 255
        
        bit_planes.append(plane)
    
    return bit_planes


# ---------------------------------
# Function to merge bit planes
# ---------------------------------
def merge_all_planes(bit_planes):
    
    merged = np.zeros_like(bit_planes[0])
    
    for i in range(8):
        merged += (bit_planes[i] // 255) << i
    
    return merged
    

def merge_partial_plane(bit_planes, indices):

    # Start with black image
    merged = np.zeros_like(bit_planes[0])

    # Add only selected planes
    for i in indices:
        merged += (bit_planes[i] // 255) << i

    return merged

    

# ---------------------------------
# Main function
# ---------------------------------
def main():
    
    # Load grayscale image
    image = cv2.imread("images/image1.jpg", 0)  # Replace with your image name
    
    # Get bit planes
    bit_planes = get_bit_planes(image)
    
    # Merge all bit planes
    merged_image = merge_all_planes(bit_planes)
    # Merge partial bit planes (0, 2, 4) and 1, 5, 6, 7
    partial_1 = merge_partial_plane(bit_planes, [0, 2, 4])
    partial_2 = merge_partial_plane(bit_planes, [1, 5, 6, 7])
    # Display results
    plt.figure(figsize=(12, 8))
    
    # Original
    plt.subplot(4, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    plt.axis("off")
    
    
    # Show 8 bit planes
    for i in range(8):
        plt.subplot(4, 4, i + 2)
        plt.imshow(bit_planes[i], cmap='gray')
        plt.title("Bit Plane " + str(i))
        plt.axis("off")
    
    
    # Merged image
    plt.subplot(4, 4, 11)
    plt.imshow(merged_image, cmap='gray')
    plt.title("Merged")
    plt.axis("off")
    # Partial 1
    plt.subplot(4, 4, 12)
    plt.imshow(partial_1, cmap='gray')
    plt.title("Merged: 0 2 4")
    plt.axis("off")
    # partial 2
    plt.subplot(4, 4, 13)
    plt.imshow(partial_2, cmap='gray')
    plt.title("Merged: 1,5,6,7")
    plt.axis("off")
    
    
    plt.tight_layout()
    import os
    os.makedirs("output_images", exist_ok=True)
    plt.savefig("output_images/bit_planes.png")


# Run program
if __name__ == "__main__":
    main()
