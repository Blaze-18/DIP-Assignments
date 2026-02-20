# ==========================================================
# IMAGE PROCESSING + VISUALIZATION USING MATPLOTLIB
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# 1. SPLIT IMAGE INTO S x S PARTS
# ----------------------------------------------------------
def split_into_parts(image, s):
    
    rows = np.array_split(image, s, axis=0)
    grids = []
    
    for row in rows:
        cols = np.array_split(row, s, axis=1)
        grids.append(cols)
    
    return grids


# ----------------------------------------------------------
# 2. MERGE PARTS BACK
# ----------------------------------------------------------
def merge_parts(grids):
    
    rows = []
    for row in grids:
        rows.append(np.hstack(row))
    
    final_image = np.vstack(rows)
    return final_image


# ----------------------------------------------------------
# 3. APPLY HE TO EACH PART
# ----------------------------------------------------------
def apply_he_to_parts(grids):
    
    s = len(grids)
    
    for i in range(s):
        for j in range(s):
            grids[i][j] = cv2.equalizeHist(grids[i][j])
    
    return grids


# ----------------------------------------------------------
# 4. APPLY AHE
# ----------------------------------------------------------
def apply_ahe(image):
    
    ahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
    return ahe.apply(image)


# ----------------------------------------------------------
# 5. APPLY CLAHE
# ----------------------------------------------------------
def apply_clahe(image, clip_value):
    
    clahe = cv2.createCLAHE(clipLimit=clip_value,
                            tileGridSize=(8,8))
    return clahe.apply(image)


# ----------------------------------------------------------
# 6. VISUALIZE IMAGES
# ----------------------------------------------------------
def visualize_images(images, titles):
    
    n = len(images)
    
    plt.figure(figsize=(15, 8))
    
    for i in range(n):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("output_images/image_plots_q11.png")
    print("Images saved")


# ----------------------------------------------------------
# 7. PLOT AND SAVE HISTOGRAMS
# ----------------------------------------------------------
def plot_and_save_histograms(images, titles):
    
    n = len(images)
    
    plt.figure(figsize=(15, 8))
    
    for i in range(n):
        plt.subplot(2, 3, i+1)
        plt.hist(images[i].ravel(), bins=256)
        plt.title(titles[i])
    
    plt.tight_layout()
    plt.savefig("output_images/histogram_plots_q11.png")
    print("Histograms saved")


# ----------------------------------------------------------
# 8. MAIN FUNCTION
# ----------------------------------------------------------
def main():
    
    image = cv2.imread("images/image2.png", 0)
    
    if image is None:
        print("Image not found.")
        return
    
    s = 3   # Split into 3x3 parts
    
    # Step 1: Split
    grids = split_into_parts(image, s)
    
    # Step 2: HE on each part
    he_grids = apply_he_to_parts(grids)
    
    # Step 3: Merge
    he_merged = merge_parts(he_grids)
    
    # Step 4: AHE
    ahe_image = apply_ahe(image)
    
    # Step 5: CLAHE
    clahe_low = apply_clahe(image, 2.0)
    clahe_high = apply_clahe(image, 8.0)
    
    
    # ------------------------------------------------------
    # Create image and title arrays
    # ------------------------------------------------------
    images = [
        image,
        he_merged,
        ahe_image,
        clahe_low,
        clahe_high
    ]
    
    titles = [
        "Original Image",
        "HE on Each Part (Merged)",
        "Adaptive Histogram Equalization (AHE)",
        "CLAHE - Clip 2.0",
        "CLAHE - Clip 8.0"
    ]
    
    
    # Visualize images
    visualize_images(images, titles)
    
    # Plot and save histograms
    plot_and_save_histograms(images, titles)


# ----------------------------------------------------------
# RUN PROGRAM
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
