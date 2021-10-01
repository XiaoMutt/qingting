import os

from core.ml.dqn.archiver.archiver import Archiver
from core.ml.dqn.environment.image_state import ImageState
from core.ml.dqn.recorder.episode_player import EpisodePlayer
from core.ml.dqn.recorder.episode_recorder import EpisodeRecorder


class TrainingSaver(Archiver):
    def __init__(self, output_folder: str, render: bool = False):
        super(TrainingSaver, self).__init__(output_folder, 20)
        self.episode_player = EpisodePlayer() if render else None
        self.__recorder = None
        self.__t = None

    def _render(self, t, frame, action, q_vals):
        if self.episode_player:
            self.episode_player.render(t, frame, action, q_vals)

    @property
    def is_recording(self):
        return self.__recorder is not None

    def start(self, t: int, image_state: ImageState):
        assert self.is_recording is False, "Recording already started"
        self.__recorder = EpisodeRecorder(image_state)
        self.__t = t

    def add(self, t, frame, action, q_vals):
        if self.is_recording:
            self.__recorder.add(action, q_vals)
        self._render(t, frame, action, q_vals)

    def save(self):
        if self.is_recording:
            path = os.path.join(self.output_folder, f'training_at_t{self.__t:09d}.record')
            self.__recorder.save(path)
            self._register_archived(path)
            self.__recorder = None
            self.__t = None
