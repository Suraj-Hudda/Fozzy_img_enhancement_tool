import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image and convert it to grayscale
image = cv2.imread('house.tiff')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image)

# Define a 3x3 averaging kernel
kernel = np.ones((3, 3), np.float32) / 9

# Apply convolution to calculate the mean of each 3x3 neighborhood
mean_3x3 = cv2.filter2D(gray_image, -1, kernel)
print("mean_wali:",mean_3x3)

# Plotting the original image and its histogram
plt.figure(figsize=(12, 6))

# Original Grayscale Image
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

# Histogram of Original Grayscale Image
plt.subplot(2, 2, 2)
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.title("Histogram of Original Image")

# Mean Filtered Image
plt.subplot(2, 2, 3)
plt.imshow(mean_3x3, cmap='gray')
plt.title("Mean Filtered Image")
plt.axis('off')

# Histogram of Mean Filtered Image
plt.subplot(2, 2, 4)
plt.hist(mean_3x3.ravel(), 256, [0, 256])
plt.title("Histogram of Mean Filtered Image")

plt.tight_layout()
plt.show()
