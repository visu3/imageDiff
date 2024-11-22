import cv2
import functions
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Read the benchmark image (undamaged)
image_new_container = cv2.imread('./new.jpg')  # Load your original image

# Normalize pixel values to [0, 1]
normalized_image_new_container = image_new_container.astype('float32') / 255.0

# Resize to desired dimensions (for example, 400x300)
resized_image_new_container = cv2.resize(normalized_image_new_container, (400, 300))

# Reduce noise using Gaussian Blur
denoised_image_new_container = cv2.GaussianBlur(resized_image_new_container, (5, 5), 0)

# Read the benchmark image (undamaged)
image_old_container = cv2.imread('./old.jpg')  # Load your original image

# Normalize pixel values to [0, 1]
normalized_image_old_container = image_old_container.astype('float32') / 255.0

# Resize to desired dimensions (for example, 400x300)
resized_image_old_container = cv2.resize(normalized_image_old_container, (400, 300))

# Reduce noise using Gaussian Blur
denoised_image_old_container = cv2.GaussianBlur(resized_image_old_container, (5, 5), 0)


# Convert to grayscale
gray1 = cv2.cvtColor(denoised_image_new_container, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(denoised_image_old_container, cv2.COLOR_BGR2GRAY)


# Compute absolute difference
difference = cv2.absdiff(gray1, gray2)

# Apply a binary threshold to get a binary image
_, thresholded = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

# Dilate the thresholded image to enhance features
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(thresholded, kernel, iterations=1)

# # Display the images
# cv2.imshow('Benchmark Image', image1)
# cv2.imshow('Test Image', image2)
# cv2.imshow('Difference', difference)
# cv2.imshow('Thresholded', thresholded)
# cv2.imshow('Dilated', dilated)

# Ensure the images are the same size
if gray1.shape != gray2.shape:
    gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

# Compute SSIM between the two images
ssim_index, diff = ssim(gray1, gray2, data_range=gray1.max() - gray1.min(),full=True)

# Print the SSIM index
print("SSIM Index:", ssim_index)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(denoised_image_new_container, cv2.COLOR_BGR2RGB))
plt.title('Baseline Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(denoised_image_old_container, cv2.COLOR_BGR2RGB))
plt.title('Damaged Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(diff, cmap='gray')
plt.title('Difference Image')
plt.axis('off')

plt.show()

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()