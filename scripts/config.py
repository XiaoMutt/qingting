import os
from datetime import datetime

from core.basis.immutable import Immutable
from core.basis.logger import setup_logger

TRAIN_DEV_TEST_RATIO = (0.8, 0.1, 0.1)
TRAIN_DEV_TEST_SPLIT_RANDOM_SEED = 1276

MNIST_TRAIN_DATA_PATH = "/Volumes/XiaoSSD/data/raw/mnist/train.hdf5"
MNIST_TEST_DATA_PATH = "/Volumes/XiaoSSD/data/raw/mnist/test.hdf5"
USPS_DATA_PATH = r'/Volumes/XiaoSSD/data/raw/usps.h5'
ARDIS_TRAIN_IMAGE_DATA_PATH = '/Volumes/XiaoSSD/data/raw/ARDIS_DATASET_IV/ARDIS_train_2828.csv'
ARDIS_TRAIN_LABEL_DATA_PATH = '/Volumes/XiaoSSD/data/raw/ARDIS_DATASET_IV/ARDIS_train_labels.csv'
ARDIS_TEST_IMAGE_DATA_PATH = '/Volumes/XiaoSSD/data/raw/ARDIS_DATASET_IV/ARDIS_test_2828.csv'
ARDIS_TEST_LABEL_DATA_PATH = '/Volumes/XiaoSSD/data/raw/ARDIS_DATASET_IV/ARDIS_test_labels.csv'

FLICKR30K_IMAGE_PATH = "/Volumes/XiaoSSD/data/raw/flickr30k_images/flickr30k_images"

PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))

DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

COLLECTED_BACKGROUNDS_PATH = os.path.join(DATA_FOLDER, "backgrounds.dat")
COLLECTED_DIGIT_IMAGES_PATH = os.path.join(DATA_FOLDER, "digit_images.dat")
COLLECTED_DIGIT_LABELS_PATH = os.path.join(DATA_FOLDER, "digit_labels.dat")

OUTPUT_FOLDER = os.path.join(PROJECT_FOLDER, "outputs")
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)


class TRAINING(Immutable):
    def __init__(self, ID=None):
        self.ID = ID if ID is not None else str(int(datetime.now().timestamp()))

        self.OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, self.ID)
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)

        self.MODEL_SAVE_FOLDER = os.path.join(self.OUTPUT_FOLDER, "models")
        if not os.path.exists(self.MODEL_SAVE_FOLDER):
            os.mkdir(self.MODEL_SAVE_FOLDER)

        self.LOG_SAVE_FOLDER = os.path.join(self.OUTPUT_FOLDER, "logs")
        if not os.path.exists(self.LOG_SAVE_FOLDER):
            os.mkdir(self.LOG_SAVE_FOLDER)
        self.LOGGER = setup_logger(self.LOG_SAVE_FOLDER, 'training_log')

        self.RECORD_SAVE_FOLDER = os.path.join(self.OUTPUT_FOLDER, 'records')
        if not os.path.exists(self.RECORD_SAVE_FOLDER):
            os.mkdir(self.RECORD_SAVE_FOLDER)
