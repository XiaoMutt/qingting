import random
import typing as tp

import numpy as np
import torch

from core.etl.image_composer import ImageComposer


class InnImageComposer(ImageComposer):
    def compose_one(self) -> tp.Tuple[int, np.ndarray, np.ndarray]:
        """
        Compose an image with num of digits on it
        :return: a tuple of
            digit_label (int),
            digit image with background (np.ndarray: H, W),
            digit probability cube (np.ndarray: 11, H, W). the 11 channels are 0-9 digit and not a digit probability
        """
        bg_imar = self.bgh()

        if random.random() < 1 / 11:
            digit_imar = np.zeros(self.digit_size, np.uint8)
            digit_label = 10  # empty: no number
        else:
            digit_index = self.dih.get_random_index()
            digit_imar = self.dih.get(digit_index)
            digit_label = self.dlh.get(digit_index)

        dh, dw = self.dih.size
        bgh, bgw = self.bgh.size

        tu = np.random.randint(0, bgh - dh)  # top upper
        tl = np.random.randint(0, bgw - dw)  # top left

        loc = bg_imar[tu:tu + dh, tl:tl + dw]
        img_light = np.quantile(loc, 0.9)
        if img_light > 192:
            # use black
            digit = (255 - digit_imar) * (digit_imar > 0)
        else:
            # use white
            digit = digit_imar * (digit_imar > 0)
        digit_with_bg_imar = (bg_imar[tu:tu + dh, tl:tl + dw] * (digit_imar == 0) + digit)
        digit_with_bg_imar = (digit_with_bg_imar / 255).astype(np.float32)

        pgrid = np.ones(self.dih.size, dtype=np.float32)
        pgrid[digit_imar == 0] = -1

        return digit_with_bg_imar, digit_label, pgrid

    def compose(self, batch_size: int):
        data_batch = []
        label_batch = []
        pgrid_batch = []

        for _ in range(batch_size):
            data, label, pgrid = self.compose_one()
            data_batch.append([data])  # 1xHxW
            label_batch.append(label)
            pgrid_batch.append([pgrid])  # 1xHxW

        data_batch = torch.tensor(data_batch)
        label_batch = torch.tensor(label_batch)
        pgrid_batch = torch.tensor(pgrid_batch)
        return data_batch, label_batch, pgrid_batch
