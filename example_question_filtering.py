"""
    Task		Target Visual Features
1. The Blurry Scan		Sharpen the scaned text so it is readable.
2. Industrial Cracks	Find cracks of metal grains with thin, dark lines.
3. Night Lane Tracking	Look for dark roads with "noisy" or pixelated surfaces.
4. Medical Detailer	    Find images where bone fractures appear as sharp dark gaps.
Image Visuals for Your Tasks
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_kernel(kernel_type):
    if kernel_type == "sobel":
        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        sobel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        return sobel_x, sobel_y
        
        return prewitt_x, preewitt_y
    if kernel_type == "scharr":
        scharr_x = np.array([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ])
        scharr_y = np.array([
            [-3, -10, 3],
            [0, 0, 0],
            [3, 10, 3]
        ])
        
        return scharr_x, scharr_y
        
    if kernel_type == "sharpen":
        return np.array([
            [0,-1, 0],
            [-1, 5, -1],
            [-0, -1, 0]
        ])
    
    if kernel_type == "laplace":
        return np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
    
    if kernel_type == "box_blur":
        return np.ones((5,5)) / 25
    


def apply_filtering(img, kernel):
    img_h, img_w = img.shape
    k_h, k_w = kernel.shape
    pad = k_h // 2
    
    padded = np.pad(img, pad, mode='reflect')
    
    result = np.zeros_like(img, dtype=np.float32)
    
    for i in range(img_h):
        for j in range(img_w):
            roi = padded[i:i+k_h, j:j+k_w]
            result[i,j] = np.sum(roi * kernel)
    
    result = np.abs(result)
    
    result = np.clip(result, 0, 255)
    
    return result

def task1_blurry_scan(img, kernel):
    # uses normal sharpening kernel
    return apply_filtering(img, kernel)

def task2_metal_cracks(img, kernel):
    # uses laplace
    return apply_filtering(img, kernel)

def task3_night_lane(img, kernel1, kernel2):
    # uses box_blur and then sobel_x
    noise_remove = apply_filtering(img, kernel1)
    detect_lane = apply_filtering(img, kernel2)
    return detect_lane
    
def task4_hand_fracture(img, kernel):
    # uses laplace
    return apply_filtering(img, kernel)
    
def save_img(images, titles):
    
    plt.figure(figsize=(14,8))
    for i in range(len(images)):
        plt.subplot(2,4, i+1)
        plt.axis("off")
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
    plt.savefig("output/filtering_tasks.png")
    print("Figure saved")

def main():
    blurry_scan = cv2.imread("images/blurry_ocr.png", 0)
    metal_cracks = cv2.imread("images/metal_crack.png", 0)
    night_road = cv2.imread("images/night_road.png", 0)
    fracture = cv2.imread("images/fracture.png", 0)
    print("Images Read")
    sobel_x, sobel_y = get_kernel("sobel")
    scharr_x, scharr_y = get_kernel("scharr")
    sharpen = get_kernel("sharpen")
    laplace = get_kernel("laplace")
    box_blur = get_kernel("box_blur")
    
    output1 = task1_blurry_scan(blurry_scan, sharpen)
    output2 =  task2_metal_cracks(metal_cracks, laplace)
    output3 = task3_night_lane(night_road, box_blur, sobel_x)
    output4 = task4_hand_fracture(fracture, laplace)
    
    images = [
        blurry_scan, metal_cracks, night_road, fracture,
        output1, output2, output3, output4
    ]
    titles = [
        "Original Blurry scan", "Original Metal Crack", "Original Night lane",
        "Original Fracture", "Output 1", "Output 2", "Output 3", "Output 4"
    ]
    
    save_img(images, titles)
    print("Images Saved")
    
    
if __name__ == "__main__":
    main()
    
    
