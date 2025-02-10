import cv2
import pytesseract
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR accuracy.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # Resize the image (optional, but helpful for consistency)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction using Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Optional: Deskew the image
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def extract_text(image_path):
    """
    Extract text from the preprocessed image using Tesseract OCR.
    """
    # Preprocess the image
    processed_img = preprocess_image(image_path)

    # Use Tesseract to extract text
    custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode (OEM) and Page Segmentation Mode (PSM)
    text = pytesseract.image_to_string(processed_img, config=custom_config, lang='eng')

    return text

# Example usage:
if __name__ == "__main__":
    image_path = "data/raw/sample_invoice.jpg"
    text = extract_text(image_path)
    print("Extracted Text:\n", text)