import os
import cv2
import dlib
from tqdm import tqdm
import numpy as np
import collections as col
from imutils.face_utils.helpers import shape_to_np
from utils import utils, transform
import config


def detect_face_from_file(filename, verbose=False):
    frame_idx = 0
    # Generator object
    frame_gen = utils.load_video(filename, verbose)

    # Load the detector for detecting the face
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.PREDICTOR)
    mean_face = np.load(config.MEAN_FACE)

    while True:
        try:
            # BGR, numpy array
            frame, frame_amount = frame_gen.__next__()
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

        q_landmarks.append(landmarks)
        q_frame.append(frame)

        if len(q_frame) == config.WINDOW_MARGIN:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # After transformation
            trans_frame, tform = transform.warp_img(
                smoothed_landmarks[config.STABLE_PNTS_IDS, :],
                mean_face[config.STABLE_PNTS_IDS, :],
                cur_frame,
                config.STD_SIZE)

            trans_landmarks = tform(cur_landmarks)
            # Crop mouth patch
            sequence.append(transform.crop_patch(
                trans_frame,
                trans_landmarks[config.START_IDX:config.END_IDX],
                config.CROP_WIDTH // 2,
                config.CROP_HEIGHT // 2
            ))

        if frame_idx == frame_amount - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # Transform frame
                trans_frame = transform.apply_transform(
                    tform, cur_frame, config.STD_SIZE)
                # Transform landmarks
                trans_landmarks = tform(q_landmarks.popleft())
                # Crop mouth patch
                sequence.append(transform.crop_patch(
                    trans_frame,
                    trans_landmarks[config.START_IDX:config.END_IDX],
                    config.CROP_WIDTH // 2,
                    config.CROP_HEIGHT // 2
                ))

            return np.array(sequence)

        frame_idx += 1

    return None


def traverse_and_process():
    foldernames = []

    for root, dirs, files in os.walk(config.VIDEO_DIREC):
        if len(dirs) > 0:
            foldernames = sorted(dirs)

    for folder in tqdm(foldernames, desc='Folder', bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
        # Check if dst folder exists
        utils.create_path(f'{config.SAVE_DIREC}/{folder}')

        filenames = []
        for root, dirs, files in os.walk(f'{config.VIDEO_DIREC}/{folder}'):
            filenames = sorted(file.split(".")[0] for file in list(
                filter(lambda x: x != ".DS_Store", files)))

            for filename in tqdm(filenames, desc='Files ', bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
                # for filename in filenames:
                src_path = os.path.join(
                    f'{config.VIDEO_DIREC}/{folder}', filename + ".mp4")
                dst_path = os.path.join(
                    f'{config.SAVE_DIREC}/{folder}', filename + ".npz")

                # utils.check_video_length(src_path, verbose=True)
                sequence = detect_face_from_file(src_path, verbose=False)

                assert sequence is not None, f'Cannot crop from {src_path}.'
                # print(sequence.shape)
                # ... = Ellipsis
                data = transform.convert_bgr2gray(
                    sequence) if config.CONVERT_GRAY else sequence[..., ::-1]

                utils.save2npz(dst_path, data=data)


def realtime_cropper():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    assert capture.isOpened(), IOError("Cannot open webcam")
    
    # Load the detector for detecting the face
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.PREDICTOR)
    mean_face = np.load(config.MEAN_FACE)

    frame_idx = 0

    while True:
        (status, frame) = capture.read()

        if frame_idx == 0:
            q_frame, q_landmarks = col.deque(), col.deque()
            sequence = []

        # Convert frames into grayscale
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        # Use detector to find landmarks (as rectangle)
        rects = detector(gray)
        if len(rects) != 0:
            # Determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            landmarks = predictor(gray, rects[0])
            landmarks = shape_to_np(landmarks)

            q_landmarks.append(landmarks)
            q_frame.append(frame)

            if len(q_frame) == config.WINDOW_MARGIN:
                smoothed_landmarks = np.mean(q_landmarks, axis=0)
                cur_landmarks = q_landmarks.popleft()
                cur_frame = q_frame.popleft()
                # After transformation
                trans_frame, tform = transform.warp_img(
                    smoothed_landmarks[config.STABLE_PNTS_IDS, :],
                    mean_face[config.STABLE_PNTS_IDS, :],
                    cur_frame,
                    config.STD_SIZE)

                trans_landmarks = tform(cur_landmarks)
                # Crop mouth patch
                sequence.append(transform.crop_patch(
                    trans_frame,
                    trans_landmarks[config.START_IDX:config.END_IDX],
                    config.CROP_WIDTH // 2,
                    config.CROP_HEIGHT // 2
                ))

                cv2.imshow("Transformed", transform.crop_patch(
                    trans_frame,
                    trans_landmarks[config.START_IDX:config.END_IDX],
                    config.CROP_WIDTH // 2,
                    config.CROP_HEIGHT // 2
                ))

            if frame_idx == config.FRAME_AMOUNT - 1:
                while q_frame:
                    cur_frame = q_frame.popleft()
                    # Transform frame
                    trans_frame = transform.apply_transform(
                        tform, cur_frame, config.STD_SIZE)
                    # Transform landmarks
                    trans_landmarks = tform(q_landmarks.popleft())
                    # Crop mouth patch
                    sequence.append(transform.crop_patch(
                        trans_frame,
                        trans_landmarks[config.START_IDX:config.END_IDX],
                        config.CROP_WIDTH // 2,
                        config.CROP_HEIGHT // 2
                    ))

                return np.array(sequence)

            frame_idx += 1

            # cv2.imshow("Original", frame)

        if cv2.waitKey(delay=1) == 27:
            break
    
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if config.TRAVERSE_FILES:
        traverse_and_process()
    else:
        sequence = realtime_cropper()
