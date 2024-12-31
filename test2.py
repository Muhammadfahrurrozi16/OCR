import cv2
import numpy as np

def process_image(file_path):
    # Baca gambar dari file
    image = cv2.imread(file_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use thresholding to create a binary image
    # Lowering the threshold sensitivity by using a lower threshold value
    _, binary_image = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []

    # Draw bounding boxes around each contour
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((idx, x, y, w, h))
    
    # Sort bounding boxes by y-coordinate (top side of the box)
    bounding_boxes.sort(key=lambda b: b[2])

    # Draw the bounding boxes on the image
    for _, x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(image, f"{x},{y},{w},{h}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the image with bounding boxes using OpenCV
    cv2.imshow('Bounding Boxes', image)

    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the processed image to a file
    cv2.imwrite('bounding_boxes_image.png', image)

    return bounding_boxes

# Example usage
bounding_boxes = process_image('Words/no-brand_no-brand_full01.jpg')

print("Bounding boxes (x, y, w, h):")
for bbox in bounding_boxes:
    print(bbox)

