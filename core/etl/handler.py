import os
import random

import numpy as np

from core.basis.immutable import Immutable
from core.config import COLLECTED_BACKGROUND_IMAGE_SIZE, COLLECTED_DIGIT_IMAGE_SIZE


class Handler(Immutable):
    def __init__(self, size: tuple):
        self.size = np.array(size)

    def get(self, index: int):
        raise NotImplementedError

    def get_random_index(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class FileHandler(Handler):
    def __init__(self, size: tuple, filepath: str, split: tuple, seed: int, dataset: str):
        super(FileHandler, self).__init__(size)
        self.offset = int(np.prod(size))
        self.n = os.path.getsize(filepath) // self.offset
        indices = list(range(self.n))
        random.Random(seed).shuffle(indices)
        split = np.array(split)
        split_ratio = np.cumsum(split) / np.sum(split)
        train_boundary = int(split_ratio[0] * self.n)
        dev_boundary = int(split_ratio[1] * self.n)
        if dataset == "train":
            self.indices = indices[:train_boundary]
        elif dataset == "dev":
            self.indices = indices[train_boundary:dev_boundary]
        elif dataset == "test":
            self.indices = indices[dev_boundary:]
        else:
            raise Exception(f"unknown dataset: {dataset}")
        self.reader = open(filepath, 'rb')

    def get(self, index) -> np.ndarray:
        if index >= self.n:
            raise Exception(f"index overflow: maximum {self.n - 1} but requested {index}")
        self.reader.seek(index * self.offset)
        data = self.reader.read(self.offset)
        tmp = np.frombuffer(data, dtype=np.uint8)
        res = np.array(tmp.reshape(self.size))  # create new editable array
        return res

    def get_random_index(self) -> int:
        return random.choice(self.indices)

    def close(self):
        self.reader.close()

    def __del__(self):
        if not self.reader.closed:
            self.close()

    def __call__(self, *args, **kwargs):
        """
        Return a random one
        :param args:
        :param kwargs:
        :return:
        """

        return self.get(index=self.get_random_index())


class CollectedBackgroundImageHandler(FileHandler):
    def __init__(self, filepath: str, split: tuple, seed: int, dataset: str):
        super(CollectedBackgroundImageHandler, self).__init__(
            COLLECTED_BACKGROUND_IMAGE_SIZE,
            filepath, split, seed, dataset)


class CollectedDigitImageHandler(FileHandler):
    def __init__(self, filepath: str, split: tuple, seed: int, dataset: str):
        super(CollectedDigitImageHandler, self).__init__(
            COLLECTED_DIGIT_IMAGE_SIZE,
            filepath, split, seed, dataset)


class CollectedLabelDataHandler(FileHandler):
    def __init__(self, filepath: str, split: tuple, seed: int, dataset: str):
        super(CollectedLabelDataHandler, self).__init__(
            (1,),
            filepath, split, seed, dataset
        )

    def get(self, index) -> int:
        res = super(CollectedLabelDataHandler, self).get(index)
        res = res.item()
        return res


class BlankBackgroundImageHandler(Handler):
    def __init__(self):
        super(BlankBackgroundImageHandler, self).__init__(COLLECTED_BACKGROUND_IMAGE_SIZE)

    def get(self, index: int):
        return np.zeros(COLLECTED_BACKGROUND_IMAGE_SIZE)

    def get_random_index(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return np.zeros(COLLECTED_BACKGROUND_IMAGE_SIZE)
