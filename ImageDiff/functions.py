def preprocess_image(image):
   
    # # Load the image
    # image = cv2.imread(image_path)

    # Normalize pixel values to [0, 1]
    normalized_image = image.astype('float32') / 255.0
    print("Normalize pixel values to [0, 1]")

    # Resize to desired dimensions (for example, 400x300)
    resized_image = cv2.resize(normalized_image, (400, 300))
    print("Resize to desired dimensions (for example, 400x300))")

    # Reduce noise using Gaussian Blur
    denoised_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    print("denoised image")

    print('preprocessed successfully!')
    return denoised_image