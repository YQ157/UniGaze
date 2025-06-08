import os
import os.path as osp
import cv2
import numpy as np

from util_func import write_error

def get_landmarks_from_image(image, img_name, fa, error_log, resize_factor=0.25):
    image_resize = cv2.resize(image, dsize=None, fx=resize_factor, fy=resize_factor, interpolation = cv2.INTER_AREA)
    image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(image_resize)

    if preds is not None:
        ## each pred is a detected face
        landmarks = preds[0] # array (68,2)
        landmarks /= resize_factor
        # Head Pose Estimation
        landmarks = np.asarray(landmarks)
        if len(preds)>1:
            write_error( f'{img_name} has more than one face detected', error_log)
        
        return landmarks
    else:
        print(f'{img_name} has no face detected')
        write_error( f'{img_name} has no face detected', error_log)
        return None