import typing as tp

import numpy as np

from core.etl.image_composer import ImageComposer


class DqnImageComposer(ImageComposer):

    def compose(self, num: int) -> tp.Tuple[np.ndarray, list, list]:
        """
        Compose an image with num of digits on it
        :param num: num of digits
        :return: image ndarray, list of digit regions, list of digit image arrays
        """
        regions = []
        segments = []
        bg_imar = self.bgh()
        while len(regions) < num:
            digit_imar = self.dih()

            dh, dw = self.digit_size
            bgh, bgw = self.image_size

            tr = np.random.randint(0, bgh - dh)  # top left row
            tc = np.random.randint(0, bgw - dw)  # top left col

            region = np.array([[tr, tc], [tr + dh, tc + dw]], dtype=np.int32)

            if self._overlap(region, regions):
                # overlap reject
                continue

            loc = bg_imar[tr:tr + dh, tc:tc + dw]
            img_light = np.quantile(loc, 0.9)
            if img_light > 192:
                # use black
                digit = (255 - digit_imar) * (digit_imar > 0)
            else:
                # use white
                digit = digit_imar * (digit_imar > 0)
            bg_imar[tr:tr + dh, tc:tc + dw] = bg_imar[tr:tr + dh, tc:tc + dw] * (digit_imar == 0) + digit
            regions.append(region)
            segments.append(digit_imar)
        bg_imar = (bg_imar / 255).astype(np.float32)
        return bg_imar, regions, segments
