from collections import defaultdict, deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter(object):
    def __init__(self, output_folder: str, max_queue: int = 1000000, maxlen: int = 100):
        self._summary_writer = SummaryWriter(output_folder, max_queue=max_queue)
        self._dict = defaultdict(lambda: deque(maxlen=maxlen))

    def __getitem__(self, item):
        return self._dict[item]

    def output(self, t) -> dict:
        res = dict()
        for k, v in self._dict.items():
            res[f'Avg_{k}'] = np.mean(v)
            res[f'Std_{k}'] = np.std(v)

        for k, v in res.items():
            self._summary_writer.add_scalar(k, v, t)
        return res
