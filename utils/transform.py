import cv2
import numpy as np
from skimage import transform as tf


# Landmark interpolation
def linear_interpolation(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + \
            idx / float(stop_idx - start_idx) * delta
    return landmarks


# Face transformation
def warp_img(src, dst, img, std_size):
    # Find the transformation matrix
    tform = tf.estimate_transform('similarity', src, dst)
    # Warp the frame image
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    # Output from warp is double image. Value range: [0, 1]
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    # Warp the frame image
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    # Output from warp is double image. Value range: [0, 1]
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped


def crop_patch(img, landmarks, width, height, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_x - width < 0:
        center_x = width
    if center_x - height < 0 - threshold:
        raise Exception('Too much bias in width')

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('Too much bias in height')

    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')

    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                             int(round(center_x) - round(width)): int(round(center_x) + round(width))])

    return cutted_img


# Convert BGR to GRAY
def convert_bgr2gray(data):
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)
