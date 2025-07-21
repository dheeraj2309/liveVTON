import cv2
import numpy as np

def resize_with_aspect(image, target_size=(256, 192), pad_color=(0, 0, 0)):
    """
    Resize image to target_size while maintaining aspect ratio and padding.
    target_size: (height, width)
    """
    target_h, target_w = target_size
    h, w = image.shape[:2]

    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with the scale
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded image
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                 borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded

# Example usage
img = cv2.imread("cropped_person.jpg")
output = resize_with_aspect(img, target_size=(256, 192))
cv2.imwrite("resized.jpg", output)
