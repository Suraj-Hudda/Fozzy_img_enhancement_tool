import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

# # Step 1: Load the original image
# image = cv2.imread('house.tiff', cv2.IMREAD_GRAYSCALE)
# img=cv2.imread('house.tiff')
def grayscale_enhance(image):
    # Step 2: Normalize the pixel values (0 to 1)
    normalized_image = image / 255.0

    # Step 3: Fuzzy intensification process (sample formula applied)
    def fuzzy_intensification(pixel_value, alpha=2):
        if 0 < pixel_value <= 0.5:
            return 2 * (pixel_value ** alpha)
        elif 0.5 < pixel_value < 1:
            return 1 - 2 * ((1 - pixel_value) ** alpha)
        else:
            return pixel_value

    # Apply fuzzy intensification
    intensified_image = np.vectorize(fuzzy_intensification)(normalized_image)

    # Step 4: Defuzzify (convert back to pixel values 0 to 255)
    defuzzified_image = (intensified_image * 255).astype(np.uint8)

    # Step 5: Brightness preservation
    original_brightness = np.mean(image)
    enhanced_brightness = np.mean(defuzzified_image)
    brightness_preserved_image = defuzzified_image * (original_brightness / enhanced_brightness)
    brightness_preserved_image = np.clip(brightness_preserved_image, 0, 255).astype(np.uint8)


    return brightness_preserved_image













# # Step 6: Display both original and enhanced images
# plt.figure(figsize=(10, 5))

# # Display the original image
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.title("Original Image")

# # Display the enhanced image
# plt.subplot(1, 2, 2)
# plt.imshow(brightness_preserved_image, cmap='gray')
# plt.title("Enhanced Image")

# plt.show()
