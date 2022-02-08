import os
import re
import cv2
import numpy as np


# IO utilities
def read_txt_lines(filepath):
    assert os.path.isfile(filepath), f"File not found: {filepath}"
    with open(filepath, 'r') as f:
        contents = f.read().splitlines()
    return contents


def save2npz(filename, data=None):
    assert data is not None, f"data is {data}"
    if not os.path.exists(os.path.dirname(filename)):
        # Create the directory if it doesn't exist'
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)


def load_video(filename, display=False):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while cap.isOpened():
        # BGR
        ret, frame = cap.read()
        # FPS = 1 / desired FPS
        FPS = 1 / 60
        FPS_MS = int(FPS * 1000)

        if ret:
            if display:
                cv2.imshow("Frame", frame)
                cv2.waitKey(FPS_MS)
                if cv2.waitKey(delay=1) == 27:
                    break
            yield frame
        else:
            break
    cap.release()