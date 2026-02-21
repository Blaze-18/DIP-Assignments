import cv2
import matplotlib.pyplot as plt
import numpy as np

def auto_filtering(img, kernel):    
    filtered = cv2.filter2D(img, -1, kernel);
    return filtered.astype(np.uint8)

def manual_filtering(img, kernel):
    
    img_h, img_w = img.shape
    k_h, k_w = kernel.shape
    
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    padded = np.pad(img, pad_h, mode='reflect')
    
    output = np.zeros_like(img, dtype=np.uint8)
    
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i + k_h, j: j + k_w]
            output[i,j] = np.sum(region * kernel)
    
    output = np.clip(output, 0, 255)
    
    return output
    
def show_img(images, titles):

    plt.figure(figsize=(10,5))
    for i in range(len(images)):
        plt.subplot(3,2,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        plt.title(titles[i])
        
    plt.savefig("test_output/spatial_filters.png")

def main():
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    # use your own kernel or different kernels 
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    
    img = cv2.imread("images/image2.png", 0)
    
    sobel_manual = manual_filtering(img, sobel_x)
    sobel_auto = auto_filtering(img, sobel_x)
    
    prewitt_manual =  manual_filtering(img, prewitt_x)
    prewitt_auto = auto_filtering(img, prewitt_x)
    
    images = [img, sobel_manual, sobel_auto, prewitt_manual, prewitt_auto]
    titles = ["Original", "Manual Sobel X", 
            "Auto Sobel X", "Manual Prewitt x", 
            "Auto Prewitt X"
        ]
    
    show_img(images, titles)


if __name__ == "__main__":
    main()
    
    
    
    
