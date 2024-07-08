import cv2

# Directory path
image_directory = "./openCv/path/"
image_path = image_directory + "image.jpg"

# Task 1: Loading and Displaying an Image
original_image = cv2.imread(image_path)

# Check if the image is loaded successfully
if original_image is None:
    raise FileNotFoundError(f"Error: Unable to load the image at {image_path}")

cv2.imshow("Original Image", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Task 2: Convert Image to Grayscale
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Task 3: Simple Image Operations (Crop and Rotate)
# Cut the image in half
height, width, _ = original_image.shape
half_width = width // 2
cropped_image = original_image[:, :half_width]

# Rotate the cropped image by 90 degrees clockwise
rotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Task 4: Saving an Image
cv2.imwrite(image_directory + "grayscale_image.jpg", grayscale_image)
cv2.imwrite(image_directory + "rotated_image.jpg", rotated_image)
cv2.imwrite(image_directory + "cropped_image.jpg", cropped_image)

# Task 5: Edge Detection
edges = cv2.Canny(grayscale_image, 75, 200)
cv2.imshow("Edge Detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Task 6: Simple Face Detection
# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the cascade classifier is loaded successfully
if face_cascade.empty():
    raise FileNotFoundError("Error: Unable to load the Haar cascade classifier.")

# Convert to grayscale for face detection
gray_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_original_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

# Draw rectangles around the detected faces
for i, (x, y, w, h) in enumerate(faces):
    face_roi = original_image[y:y+h, x:x+w]
    cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(image_directory + f"face_{i}.jpg", face_roi)
  
cv2.imshow("Face Detection", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(image_directory + "detection_image.jpg", original_image)

