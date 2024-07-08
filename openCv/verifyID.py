import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to preprocess the image for better OCR results
def preprocess_image(image_path):
    # Read the image using cv2
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_image = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return processed_image

# Function to perform OCR on the preprocessed image
def perform_ocr(image):
    # Use pytesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(image, config='--psm 6')

    return extracted_text

# Function to verify the country of origin based on extracted text
def verify_country_of_origin(extracted_text, declared_country):
    # Implement your logic to check compatibility
    # You can use simple pattern matching or more advanced text analysis techniques

    # For simplicity, let's assume that the extracted text contains the country name
    # Check if the declared country matches the extracted country
    if declared_country.lower() in extracted_text.lower():
        return "ID OK"
    else:
        return "Invalid ID"

# Main function
def main(image_path, declared_country):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Perform OCR on the preprocessed image
    extracted_text = perform_ocr(processed_image)
    
    # Print the extracted text for debugging
    # print("Extracted Text:", extracted_text)

    # Verify the country of origin
    verification_result = verify_country_of_origin(extracted_text, declared_country)

    # Display the verification result
    print(verification_result)

# Example usage
if __name__ == "__main__":
    # Replace 'sample_id_image.jpg' with the path to your ID image
    image_path = './openCv/path/image.jpg'

    # Replace 'Brazil' with the declared country by the user
    declared_country = 'Poland'

    # Call the main function with the image path and declared country
    main(image_path, declared_country)
