from core.etl.handler import CollectedBackgroundImageHandler, CollectedDigitImageHandler, CollectedLabelDataHandler, \
    BlankBackgroundImageHandler
from scripts.config import COLLECTED_BACKGROUNDS_PATH, COLLECTED_DIGIT_IMAGES_PATH, COLLECTED_DIGIT_LABELS_PATH
from scripts.config import TRAIN_DEV_TEST_RATIO, TRAIN_DEV_TEST_SPLIT_RANDOM_SEED

train_background_image_handler = CollectedBackgroundImageHandler(
    filepath=COLLECTED_BACKGROUNDS_PATH,
    split=TRAIN_DEV_TEST_RATIO,
    seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
    dataset="train"
)
train_digit_image_handler = CollectedDigitImageHandler(filepath=COLLECTED_DIGIT_IMAGES_PATH,
                                                       split=TRAIN_DEV_TEST_RATIO,
                                                       seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
                                                       dataset="train")

train_digit_label_handler = CollectedLabelDataHandler(
    filepath=COLLECTED_DIGIT_LABELS_PATH,
    split=TRAIN_DEV_TEST_RATIO,
    seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
    dataset="train")

dev_background_image_handler = CollectedBackgroundImageHandler(
    filepath=COLLECTED_BACKGROUNDS_PATH,
    split=TRAIN_DEV_TEST_RATIO,
    seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
    dataset="dev"
)
dev_digit_image_handler = CollectedDigitImageHandler(filepath=COLLECTED_DIGIT_IMAGES_PATH,
                                                     split=TRAIN_DEV_TEST_RATIO,
                                                     seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
                                                     dataset="dev")

dev_digit_label_handler = CollectedLabelDataHandler(
    filepath=COLLECTED_DIGIT_LABELS_PATH,
    split=TRAIN_DEV_TEST_RATIO,
    seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
    dataset="test")


test_background_image_handler = CollectedBackgroundImageHandler(
    filepath=COLLECTED_BACKGROUNDS_PATH,
    split=TRAIN_DEV_TEST_RATIO,
    seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
    dataset="test"
)
test_digit_image_handler = CollectedDigitImageHandler(filepath=COLLECTED_DIGIT_IMAGES_PATH,
                                                     split=TRAIN_DEV_TEST_RATIO,
                                                     seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
                                                     dataset="dev")

test_digit_label_handler = CollectedLabelDataHandler(
    filepath=COLLECTED_DIGIT_LABELS_PATH,
    split=TRAIN_DEV_TEST_RATIO,
    seed=TRAIN_DEV_TEST_SPLIT_RANDOM_SEED,
    dataset="test")

blank_background_image_handler = BlankBackgroundImageHandler()
