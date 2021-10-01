from PIL import Image

from core.etl.handler import CollectedDigitImageHandler, CollectedBackgroundImageHandler
from core.ml.dqn.image_composer import ImageComposer
from scripts.config import COLLECTED_DIGIT_IMAGES_PATH, COLLECTED_BACKGROUNDS_PATH
from scripts.config import TRAIN_DEV_TEST_RATIO, TRAIN_DEV_TEST_SPLIT_RANDOM_SEED

bg_img_handler = CollectedBackgroundImageHandler(filepath=COLLECTED_BACKGROUNDS_PATH,
                                                 split=TRAIN_DEV_TEST_RATIO,
                                                 seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
                                                 dataset="train")
digit_img_handler = CollectedDigitImageHandler(filepath=COLLECTED_DIGIT_IMAGES_PATH,
                                               split=TRAIN_DEV_TEST_RATIO,
                                               seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
                                               dataset="train")

ic = ImageComposer(
    background_image_handler=bg_img_handler,
    digit_image_handler=digit_img_handler,
    digit_label_handler=None)

img, regions, segments = ic.compose(5)
im = Image.fromarray(img)
im.show()
