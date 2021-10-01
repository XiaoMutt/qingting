import numpy as np
from torch import Tensor

from core.utils import np2torch


class ReplayBuffer(object):
    def __init__(self, size: int, frame_num_per_state: int, frame_shape: tuple):
        """
        Basic replay buffer that stores the frames and reconstructs state at index i using the frames
        [i-frame_num_per_state + 1, ..., i-1, i]
        This replay buffer does not offer sample batch functionality

        :param size: how many frames to store
        :param frame_num_per_state: the state frame number
        :param frame_shape: the shape of each frame
        """
        if frame_num_per_state > size:
            raise Exception(f"size {size} < frame_num_per_state {frame_num_per_state}")

        self.size = size
        self.frame_num_per_state = frame_num_per_state
        self.frame_shape = frame_shape

        self._observations = np.empty((self.size, *frame_shape), dtype=np.float32)
        self._actions = np.empty((self.size,), dtype=np.uint8)
        self._rewards = np.empty((self.size,), dtype=np.float32)
        self._dones = np.empty((self.size,), dtype=np.bool_)

        self._left_history_index_boundary = 0  # inclusive
        self._right_history_index_boundary = 0  # exclusive
        self._waiting_for_feedback_frame_index = None  # None means no frame is waiting for feedback

        # used for compensate frames at starts while encoding the state
        self._empty_start_frames = [np.zeros((n, *frame_shape), dtype=np.float32)
                                    for n in range(1, self.frame_num_per_state)]

    @property
    def num_stored(self):
        return self._right_history_index_boundary - self._left_history_index_boundary

    @property
    def lb(self):
        if self._left_history_index_boundary > 0:
            # buffer looped back and got overridden already. Need to leave room for frame_num_per_state
            left_boundary = self._left_history_index_boundary + self.frame_num_per_state - 1
        else:
            left_boundary = 0
        return left_boundary

    @property
    def rb(self):
        return self._right_history_index_boundary

    def add_frame(self, frame: np.ndarray) -> Tensor:
        """
        Add frame to the replay buffer and return a state Tensor
        :param frame:
        :return: state Tensor stored in DEVICE
        """
        if self._waiting_for_feedback_frame_index is not None:
            raise Exception(
                f"Still waiting for the feedback from frame index {self._waiting_for_feedback_frame_index}. "
                f"Please add the feedback first.")

        self._observations[self._right_history_index_boundary % self.size] = frame
        self._waiting_for_feedback_frame_index = self._right_history_index_boundary

        # move cursor to next position
        self._right_history_index_boundary += 1
        self._left_history_index_boundary = max(0, self._right_history_index_boundary - self.size)
        res = self._get_encoded_state_at(self._waiting_for_feedback_frame_index)
        res = np2torch(np.array([res]))
        return res

    def add_feedback(self, action: int, reward: float, done: bool) -> None:
        """
        Add feedback of the action taken based on the previous state
        :param action: the action has been taken
        :param reward: the reward
        :param done: whether the state is done
        :return:
        """
        if self._waiting_for_feedback_frame_index is None:
            raise Exception(f"Not accepting any feedback now. Have you added the new frame?")

        index = self._waiting_for_feedback_frame_index % self.size
        self._actions[index] = action
        self._rewards[index] = reward
        self._dones[index] = done

        self._waiting_for_feedback_frame_index = None

    def _get_encoded_state_at(self, frame_history_index: int) -> np.ndarray:
        """
        Encode the frame at frame_index:
        - a frame_num_per_state of frames with the frame_index one as the last one will be taken out from the buffer
        - all the frames must be from the same episode
        - missing frames will be substituted with zero matrix

        :param frame_history_index: the frame history index.
            Must be in [left_history_index_boundary, right_history_index_boundary)
        :return: ndarray
        """
        right = self._right_history_index_boundary
        left = self._left_history_index_boundary

        stop = frame_history_index + 1
        start = max(0, stop - self.frame_num_per_state)

        if start < left or stop > right:
            raise Exception(f"The region [{start}, {stop}) by the frame_index {frame_history_index} is "
                            f"out of buffer's boundaries [{left}, {right})")

        # take care of done frames between start and frame_history_index
        for idx in range(start, frame_history_index):
            if self._dones[idx % self.size]:
                start = idx + 1

        num_missing_start_frames = self.frame_num_per_state - (stop - start)
        start = start % self.size
        stop = stop % self.size

        # extract frames
        if start < stop:
            frames = self._observations[start: stop]
        else:
            frames = np.concatenate((self._observations[start:], self._observations[0:stop]), axis=0)

        # taking care of missing frames
        if num_missing_start_frames > 0:
            frames = np.concatenate((self._empty_start_frames[num_missing_start_frames - 1], frames), axis=0)

        # reshape to frame_num_per_state * n_channels x H x W
        res = frames.reshape((-1, self.frame_shape[-2], self.frame_shape[-1]))
        return res

    def rewind_to(self, frame_history_index):
        """
        Rewind the ReplayBuffer to history_index. Afterwards, if add_frame, the history after the history_index will
        be overridden.
        :param frame_history_index: if <0 then rewind backwards history_index steps, if >=0 rewind to that history_index
        :return: None
        """

        if frame_history_index < 0:
            frame_history_index = self._right_history_index_boundary + frame_history_index

        if frame_history_index < self.lb:
            raise Exception(f"frame_history_index = {frame_history_index} less than allowed boundary: {self.lb}")
        elif frame_history_index > self.rb:
            raise Exception(f"frame_history_index = {frame_history_index} greater than allowed boundary: {self.rb}")

        self._right_history_index_boundary = frame_history_index
        self._waiting_for_feedback_frame_index = None

    def sample(self, *args, **kwargs):
        raise NotImplementedError
