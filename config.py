# Settings
TRAVERSE_FILES = False

# File paths
VIDEO_DIREC = './datasets/raw'
SAVE_DIREC = './datasets/visual_data/'
LANDMARK_DIREC = './landmarks/'
PREDICTOR = "./modules/shape_predictor_68_face_landmarks_GTX.dat"
MEAN_FACE = './modules/20words_mean_face.npy'

# Cropping parameters
FRAME_AMOUNT = 120
CONVERT_GRAY = False
STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
START_IDX = 48
END_IDX = 68
CROP_WIDTH = 96
CROP_HEIGHT = 96
WINDOW_MARGIN = 12
