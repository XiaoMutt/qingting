import typing as tp

import numpy as np
from torch import Tensor

from core.ml.dqn.replay_buffer.replay_buffer import ReplayBuffer
from core.utils import np2torch


class UniformReplayBuffer(ReplayBuffer):

    def sample(self, batch_size: int) -> tp.Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample a batch from the buffer for training.
        :param batch_size: batch size to sample
        :return: a tuple of Tensor states, actions, rewards, next_states, dones.
        The first dimension of the Tensors is batch size
        """

        if self._waiting_for_feedback_frame_index is not None:
            raise Exception(f"Still waiting for the effect of frame {self._waiting_for_feedback_frame_index}")

        if batch_size > self.num_stored:
            raise Exception(f"Requested batch size {batch_size} > number of frames stored {self.num_stored}")

        picked_history_indices = self._sample_frame_indices(batch_size)

        obs_batch = np.stack([self._get_encoded_state_at(idx) for idx in picked_history_indices])
        next_obs_batch = np.stack([self._get_encoded_state_at(idx + 1) for idx in picked_history_indices])

        obs_indices = [idx % self.size for idx in picked_history_indices]
        act_batch = self._actions[obs_indices]
        rew_batch = self._rewards[obs_indices]

        done_mask_batch = self._dones[obs_indices]

        return np2torch(obs_batch), np2torch(act_batch), np2torch(rew_batch), \
               np2torch(next_obs_batch), np2torch(done_mask_batch)

    def _sample_frame_indices(self, batch_size: int) -> tp.List[int]:
        """
        Sample a set of frame history indices.
        ATTENTION:
            - the indices are frame history indices which must be converted to buffer indices by % self.size
                before extracting data from buffer.

        :param batch_size:
        :return:
        """
        res = set()
        while len(res) < batch_size:
            tmp = np.random.randint(self.lb, self.rb - 1, size=batch_size - len(res))
            res.update(tmp)
        res = list(res)
        return res
