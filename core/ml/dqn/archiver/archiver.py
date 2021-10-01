import os
from collections import deque

from core.basis.protected import Protected


class Archiver(Protected):
    def __init__(self, output_folder: str, max_num_to_keep: int):
        """
        Archiver saves file to output_folder, and keeps at most max_num_to_keep files
        :param output_folder:
        :param max_num_to_keep:
        """
        self.output_folder = output_folder
        self.history = deque()
        self.max_num_to_keep = max_num_to_keep

    def _register_archived(self, path: str):
        # remove old ones
        if len(self.history) == self.max_num_to_keep:
            to_delete = self.history.popleft()
            os.remove(to_delete)
        self.history.append(path)
