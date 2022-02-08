import os
import cv2
import dlib
from tqdm import tqdm
import numpy as np
import collections as col
import argparse
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
from imutils.face_utils.helpers import shape_to_np

import utils
import transform


class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


def detect_face(filename, display=False):
    frame_idx = 0
    # Generator object
    frame_gen = utils.load_video(filename, display)
    # Load the detector for detecting the face
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor)
    mean_face = np.load(args.mean_face)
    while True:
        try:
            # BGR, numpy array
            frame = frame_gen.__next__()
        except StopIteration:
            break

        if frame_idx == 0:
            q_frame, q_landmarks = col.deque(), col.deque()
            sequence = []

        # Convert frames into grayscale
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        # Use detector to find landmarks (as rectangle)
        rects = detector(gray)
        # Determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        landmarks = predictor(gray, rects[0])
        landmarks = shape_to_np(landmarks)

        # for (x, y) in landmarks:
        #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        # cv2.imshow("Output", frame)
        # cv2.waitKey(int(1 / 60 * 1000))
        # print(landmarks.shape)

        q_landmarks.append(landmarks)
        q_frame.append(frame)
        # print(len(q_landmarks), len(q_frame))

        if len(q_frame) == args.window_margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # print(q_landmarks, smoothed_landmarks)
            # After transformation
            trans_frame, tform = transform.warp_img(
                smoothed_landmarks[args.stablePntsIDs, :],
                mean_face[args.stablePntsIDs, :],
                cur_frame,
                args.std_size)

            trans_landmarks = tform(cur_landmarks)
            # print(trans_frame, tform, trans_landmarks)
            # Crop mouth patch
            sequence.append(transform.crop_patch(
                trans_frame,
                trans_landmarks[args.start_idx:args.end_idx],
                args.crop_width // 2,
                args.crop_height // 2
            ))

            # cv2.imshow("Transformed", transform.crop_patch(
            #     trans_frame,
            #     trans_landmarks[args.start_idx:args.end_idx],
            #     args.crop_width // 2,
            #     args.crop_height // 2
            # ))
            # cv2.waitKey(int(1 / 60 * 1000))

        if frame_idx == len(landmarks) - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # Transform frame
                trans_frame = transform.apply_transform(
                    tform, cur_frame, args.std_size)
                # Transform landmarks
                trans_landmarks = tform(q_landmarks.popleft())
                # Crop mouth patch
                sequence.append(transform.crop_patch(
                    trans_frame,
                    trans_landmarks[args.start_idx:args.end_idx],
                    args.crop_width // 2,
                    args.crop_height // 2
                ))

            # print(frame_idx, len(sequence))
            return np.array(sequence)

        frame_idx += 1
        # print(frame_idx, len(sequence), len(landmarks))
    return None


def landmarks_interpolation(landmarks):
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]

    if not valid_frames_idx:
        return None

    for idx in range(1, len(valid_frames_idx)):
        # Check if frames are continuous
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            # Interpolate missed frame
            landmarks = transform.linear_interpolation(
                landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]

    # Corner case: Keep frames at the beginning or at the end to be detected
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]

    assert len(valid_frames_idx) == len(landmarks), "Not every frame has landmarks"
    return landmarks


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = DictX({
        "video_direc": './datasets/raw',
        "landmark_direc": './landmarks/',
        "predictor": "./modules/shape_predictor_68_face_landmarks_GTX.dat",
        "save_direc": './datasets/visual_data/',
        "mean_face": './modules/20words_mean_face.npy',
        "std_size": (256, 256),
        "stablePntsIDs": [33, 36, 39, 42, 45],
        "crop_width": 96,
        "crop_height": 96,
        "start_idx": 48,
        "end_idx": 68,
        "window_margin": 12,
        "convert_gray": False,
        "testset_only": False
    })

    filenames = []

    for root, dirs, files in os.walk(args.video_direc):
        filenames = sorted(file.split(".")[0] for file in list(
            filter(lambda x: x != ".DS_Store", files)))
    
    for filename in tqdm(filenames, bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
        src_path = os.path.join(args.video_direc, filename + ".mp4")
        dst_path = os.path.join(args.save_direc, filename + ".npz")

        sequence = detect_face(src_path)
        
        assert sequence is not None, f'Cannot crop from {src_path}.'
        # print(sequence.shape)
        # ... = Ellipsis
        data = transform.convert_bgr2gray(
            sequence) if args.convert_gray else sequence[..., ::-1]
        
        utils.save2npz(dst_path, data=data)
