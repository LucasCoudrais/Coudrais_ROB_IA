import cv2
import numpy as np

def highlight_area(img, x, y, x_plus_w, y_plus_h, alpha):
    # Create a mask for the area to highlight
    mask = np.zeros_like(img)
    mask[y:y_plus_h, x:x_plus_w, :] = 255

    # Add the image and the mask, with the mask weighted to control the amount of brightness added
    highlighted_area = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)

    return highlighted_area


img = cv2.imread("my_images/img1.jpg")

# Get the coordinates of the area to highlight
x, y, x_plus_w, y_plus_h = 100, 100, 200, 200

# Highlight the area
highlighted_area = highlight_area(img, x, y, x_plus_w, y_plus_h, alpha=0.5)

# Display the image
cv2.imshow("Highlighted area", highlighted_area)
cv2.waitKey(0)
