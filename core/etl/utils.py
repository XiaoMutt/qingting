import math

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


def filter_noise(imar: np.ndarray, filter_size: int = 3):
    """
    Use median filter to filter out the random noise left in images
    :param imar: image np.ndarray
    :param filter_size: the median filter
    :return: a new filtered image np.ndarray
    """

    mask = (ndimage.median_filter(imar, size=filter_size) > 0).astype(imar.dtype)
    res = imar * mask
    return res


def crop_to_boundary(imar: np.ndarray):
    """
    Crop image to its boundary using a square
    :param imar: Image np.ndarray
    :return: a new cropped imar
    """
    h, w = imar.shape
    r0, c0, r1, c1 = 0, 0, h - 1, w - 1

    while r0 < h:
        if np.sum(imar[r0]) > 0:
            break
        r0 += 1

    while r1 > 0:
        if np.sum(imar[r1]) > 0:
            break
        r1 -= 1

    tmp = imar.T
    while c0 < w:
        if np.sum(tmp[c0]) > 0:
            break
        c0 += 1

    while c1 > 0:
        if np.sum(tmp[c1]) > 0:
            break
        c1 -= 1

    delta = r1 - r0 - (c1 - c0)
    if delta > 0:
        c1 = min(c1 + math.ceil(delta / 2), w - 1)
        c0 = c1 - (r1 - r0)
        if c0 < 0:
            c1 -= c0
            c0 = 0

    elif delta < 0:
        r1 = min(r1 + math.ceil(-delta / 2), h - 1)
        r0 = r1 - (c1 - c0)
        if r0 < 0:
            r1 -= r0
            r0 = 0
    return imar[r0:r1 + 1, c0:c1 + 1]


def resize_digit_image(imar: np.ndarray, size: int, resample=Image.BICUBIC):
    """
    resize the image to 16 by 16 pixel image
    :param imar: image np.ndarray
    :param size: size to resize to
    :param resample: resize method
    :return: a new resized Image
    """
    img = Image.fromarray(imar, mode='L')
    res = img.resize(size, resample=resample)
    return res


def gray_and_crop(img: Image, size: tuple):
    w, h = img.size
    if w < size[0] or h < size[1]:
        return None

    middlew = w // 2
    middleh = h // 2

    right = min(w, middlew + math.ceil(size[0] / 2))
    left = max(0, right - size[0])

    lower = min(h, middleh + math.ceil(size[1] / 2))
    upper = max(0, lower - size[1])

    img = img.crop((left, upper, right, lower))
    res = ImageOps.grayscale(img)
    return res



