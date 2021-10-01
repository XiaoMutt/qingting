import os

from core.ml.dqn.archiver.archiver import Archiver
from core.ml.dqn.model.dqn import Dqn


class ModelSaver(Archiver):
    def __init__(self, output_folder: str, dqn: Dqn):
        super(ModelSaver, self).__init__(output_folder, max_num_to_keep=2)
        self.dqn = dqn

    def save(self, t: int):
        # save new ones
        path = os.path.join(self.output_folder, f't{t:09d}.weights')
        self.dqn.save(path)
        self._register_archived(path)
