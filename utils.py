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


def load_video(filename, verbose=False):
    cap = cv2.VideoCapture(filename)
   
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(
            f'Filename: {filename}, Length: {length}, Width: {width}, Height: {height}, FPS: {fps}')

    while cap.isOpened():
        # BGR
        ret, frame = cap.read()

        if ret:
            yield frame, length
        else:
            break

    cap.release()
