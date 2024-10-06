import cv2
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the original color image (in RGB format)
# image = cv2.imread('anuj_color.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV format) to RGB

def rgb_enhance(image):
    # Step 2: Normalize the pixel values for each channel (0 to 1 range)
    normalized_image = image / 255.0

    # Step 3: Fuzzy intensification process for color channels
    def fuzzy_intensification(pixel_value, alpha=2):
        if 0 < pixel_value <= 0.5:
            return 2 * (pixel_value ** alpha)
        elif 0.5 < pixel_value < 1:
            return 1 - 2 * ((1 - pixel_value) ** alpha)
        else:
            return pixel_value

    # Apply fuzzy intensification to each channel separately (R, G, B)
    intensified_image = np.zeros_like(normalized_image)

    for i in range(3):  # Iterate over the R, G, B channels
        intensified_image[:, :, i] = np.vectorize(fuzzy_intensification)(normalized_image[:, :, i])

    # Step 4: Defuzzify (convert back to pixel values 0 to 255 for each channel)
    defuzzified_image = (intensified_image * 255).astype(np.uint8)

    # Step 5: Brightness preservation for each channel
    brightness_preserved_image = np.zeros_like(defuzzified_image)

    for i in range(3):
        original_brightness = np.mean(image[:, :, i])
        enhanced_brightness = np.mean(defuzzified_image[:, :, i])
        brightness_preserved_image[:, :, i] = defuzzified_image[:, :, i] * (original_brightness / enhanced_brightness)
        brightness_preserved_image[:, :, i] = np.clip(brightness_preserved_image[:, :, i], 0, 255).astype(np.uint8)
    return brightness_preserved_image





# # Step 6: Display both original and enhanced color images
# plt.figure(figsize=(10, 5))

# # Display the original image
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Original Color Image")

# # Display the enhanced image
# plt.subplot(1, 2, 2)
# plt.imshow(brightness_preserved_image)
# plt.title("Enhanced Color Image")

# plt.show()
