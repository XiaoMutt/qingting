import typing as tp

import numpy as np

from core.basis.immutable import Immutable
from core.basis.utils import iou
from core.etl.handler import Handler


class ImageComposer(Immutable):
    def __init__(self, background_image_handler: Handler, digit_image_handler: Handler,
                 digit_label_handler: tp.Optional[Handler], digit_overlapping_threshold: float = 0):
        self.bgh = background_image_handler
        self.dih = digit_image_handler
        self.dlh = digit_label_handler
        self.digit_overlapping_threshold = digit_overlapping_threshold
        self.digit_size = self.dih.size
        self.image_size = self.bgh.size

    def _overlap(self, picked: np.ndarray, exists: tp.Iterable):
        """
        Return True if the picked box has an IoU>threshold with any one in the exists
        :param picked:
        :param exists:
        :return:
        """

        for item in exists:
            if iou(picked, item) > self.digit_overlapping_threshold:
                return True
        return False

    def compose(self, *args, **kwargs):
        pass
