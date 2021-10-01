import h5py
from tqdm import tqdm

from core.config import COLLECTED_DIGIT_IMAGE_SIZE
from core.etl.utils import *
from scripts.config import *


def transform_mnist(imar):
    imar = filter_noise(imar)
    imar = crop_to_boundary(imar)
    res = resize_digit_image(imar, COLLECTED_DIGIT_IMAGE_SIZE)
    return res


def transform_usps(imar):
    imar = imar * 255
    imar = imar.astype('uint8')
    imar = imar.reshape(16, 16)
    res = resize_digit_image(imar, COLLECTED_DIGIT_IMAGE_SIZE)
    return res


def transform_ardis(imar):
    imar = filter_noise(imar)
    imar = crop_to_boundary(imar)
    res = resize_digit_image(imar, COLLECTED_DIGIT_IMAGE_SIZE)
    return res


if __name__ == '__main__':
    counter = [0]
    images = []

    image_writer = open(COLLECTED_DIGIT_IMAGES_PATH, 'wb')
    label_writer = open(COLLECTED_DIGIT_LABELS_PATH, 'wb')


    def append(fun, X, Y):
        for image, label in tqdm(zip(X, Y)):
            image = fun(image)
            b = image.tobytes()
            image_writer.write(b)
            l = np.array([label], dtype=np.uint8).tobytes()
            label_writer.write(l)


    # mnist
    with h5py.File(MNIST_TRAIN_DATA_PATH, 'r') as reader:
        X, Y = reader['image'][...], reader['label'][...]
        append(transform_mnist, X, Y)

    with h5py.File(MNIST_TEST_DATA_PATH, 'r') as reader:
        X, Y = reader['image'][...], reader['label'][...]
        append(transform_mnist, X, Y)

    # usps
    with h5py.File(USPS_DATA_PATH, 'r') as reader:
        train = reader.get('train')
        X, Y = train['data'][...], train['target'][...]
        append(transform_usps, X, Y)

        test = reader.get('test')
        X, Y = test['data'][...], test['target'][...]
        append(transform_usps, X, Y)

    # ARDIS
    X = np.loadtxt(ARDIS_TRAIN_IMAGE_DATA_PATH, dtype='uint8')
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.loadtxt(ARDIS_TRAIN_LABEL_DATA_PATH, dtype='uint8')
    Y = np.argmax(Y, axis=1)
    append(transform_ardis, X, Y)

    X = np.loadtxt(ARDIS_TEST_IMAGE_DATA_PATH, dtype='uint8')
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.loadtxt(ARDIS_TEST_LABEL_DATA_PATH, dtype='uint8')
    Y = np.argmax(Y, axis=1)
    append(transform_ardis, X, Y)

    # output
    image_writer.close()
    label_writer.close()
