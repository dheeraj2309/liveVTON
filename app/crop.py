import cv2
import mediapipe as mp
import numpy as np
import time

mp_selfie_segmentation = mp.solutions.selfie_segmentation

def crop_person(image):
    
    h, w, _ = image.shape
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentor:
        results = segmentor.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = results.segmentation_mask > 0.5
        
        # Find bounding box of the person
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return image  # No person found
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped

# Usage

img = cv2.imread("a.jpg")
start=time.time()
cropped_person = crop_person(img)
end=time.time()
print(end-start)
cv2.imwrite("cropped_person.jpg", cropped_person)
