import numpy as np
import cv2

# Create a blank image (white background)
image_width, image_height = 400, 300
baseline_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

# Draw a rectangle (representing a simple shipping container)
cv2.rectangle(baseline_image, (50, 50), (350, 200), (0, 128, 255), -1)  # Orange rectangle

# Save the baseline image
cv2.imwrite('baseline_image.jpg', baseline_image)

# Create a damaged copy
damaged_image = baseline_image.copy()

# Simulate damage (e.g., a dent or scratch)
cv2.circle(damaged_image, (200, 130), 20, (0, 0, 255), -1)  # Red circle to simulate a dent

# Save the damaged image
cv2.imwrite('damaged_image.jpg', damaged_image)

# Optionally, display the images
cv2.imshow('Baseline Image', baseline_image)
cv2.imshow('Damaged Image', damaged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
