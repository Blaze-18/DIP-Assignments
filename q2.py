# ============================================
# Program: Display 5 Shades of Colors
# Using plt.imshow()
# ============================================

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------
# Function to create 6 shades
# ---------------------------------
def shade(base_color):
    shades = []
    
    # 5 brightness levels
    # starts with black so 0.0 -> 0.2 ....
    intensity_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for f in intensity_factors:
        new_color = (
            base_color[0] * f,
            base_color[1] * f,
            base_color[2] * f
        )
        shades.append(new_color)
    
    return shades


# ---------------------------------
# Main function
# ---------------------------------
def main():
    
    colors = [
        ("White", (1, 1, 1)),
        ("Red", (1, 0, 0)),
        ("Green", (0, 1, 0)),
        ("Blue", (0, 0, 1)),
        ("Cyan", (0, 1, 1)),
        ("Magenta", (1, 0, 1)),
        ("Yellow", (1, 1, 0))
    ]
    
    rows = len(colors)
    cols = 6   # 5 shades
    
    # Create empty image array
    image = np.zeros((rows, cols, 3))
    
    # Fill the image with shades
    for i in range(rows):
        name, base_color = colors[i]
        shades = shade(base_color)
        
        for j in range(cols):
            image[i, j] = shades[j]
    
    
    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    
    plt.xticks([])
    plt.yticks([])
    
    plt.title("Five Shades of Colors from Black")
    import os
    os.makedirs("output_images", exist_ok=True)
    # Save the figure
    plt.savefig("output_images/color_shades.png")
    
    


# Run the program
if __name__ == "__main__":
    main()

