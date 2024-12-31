import cv2
import numpy as np
import copy

def draw_bounding_boxes(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around all contours
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image, bounding_boxes

def Split(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    h_proj = np.sum(thresh, axis=1)
    lines = []
    in_line = False
    for i, value in enumerate(h_proj):
        if value > 0 and not in_line:
            start = i
            in_line = True
        elif value == 0 and in_line:
            end = i
            in_line = False
            lines.append((start, end))
    if in_line:
        lines.append((start, len(h_proj)))
    
    line_boxes = []
    for start, end in lines:
        line_img = image[start:end, :]
        contours, _ = cv2.findContours(thresh[start:end, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            line_boxes.append((x, start + y, w, h))
            # Draw bounding boxes that correspond to line_boxes
            cv2.rectangle(image, (x, start + y), (x + w, start + y + h), (255, 0, 0), 2)
    
    return line_boxes

# Example usage:
image = cv2.imread('Words/5161394f598bb912d6b7ec90b09b2fa1.jpg')

# Draw initial bounding boxes around all contours
image_with_boxes, all_bounding_boxes = draw_bounding_boxes(copy.deepcopy(image))

# Split the image and draw bounding boxes around detected lines
line_boxes = Split(image)

# Save or display the images
cv2.imwrite('image_with_all_bounding_boxes.jpg', image_with_boxes)
cv2.imwrite('image_with_line_bounding_boxes.jpg', image)
