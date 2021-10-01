import typing as tp

import numpy as np
from torch import Tensor

from core.ml.dqn.replay_buffer.replay_buffer import ReplayBuffer
from core.ml.dqn.replay_buffer.segment_tree import SumSegmentTree, MinSegmentTree
from core.utils import np2torch


# TODO: this class is not finished and should not be used
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size: int, state_frame_number: int, frame_shape: tp.Tuple[int, int], alpha: float = 0.6):
        """

        :param size: how many frames to store
        :param state_frame_number: the state frame number
        :param frame_shape: the shape of each frame
        :param alpha: will be the power of priority
        """
        super(PrioritizedReplayBuffer, self).__init__(size, state_frame_number, frame_shape)
        self.tree_ptr = 0
        self.sum_tree = SumSegmentTree(self.size)
        self.min_tree = MinSegmentTree(self.size)
        self.max_priority = 1.0
        self.alpha = alpha
        self.priority_eps = 1e-6  # guarantees every transition can be sampled

        self.sampled_indices = None

    def add_feedback(self, action: int, reward: float, done: bool, info: object = None) -> None:
        super(PrioritizedReplayBuffer, self).add_feedback(action, reward, done, info)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.size

    def _sample_frame_indices(self, batch_size: int) -> tp.Tuple[tp.List[int], tp.List[int]]:
        """Sample indices based on proportions."""
        if self.num_stored == self.size:
            # buffer full and loops back
            # frame at self.cursor-1 has been replaced and should not be queried
            # ===========>=========
            #           ||||
            #         not queryable
            avoid = {(self.cursor + i) % self.size for i in range(-1, 3)}
        else:
            avoid = {self.cursor - 1}  # should not include the last frame

        avoid.add(-1)  # -1 is the result of max_idx when no element can be found
        indices = set()
        p_total = self.sum_tree.sum(0, self.num_stored - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            ceiling = np.random.uniform(a, b)
            idx = self.sum_tree.max_idx(ceiling)
            while idx in avoid or idx in indices:
                # random sample one uniformly
                idx = np.random.randint(0, self.num_stored)
            indices.add(idx)

        obs_indices = list(indices)
        next_obs_indices = [idx + 1 for idx in obs_indices]
        return obs_indices, next_obs_indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.num_stored) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.num_stored) ** (-beta)
        weight = weight / max_weight

        return weight

    def sample(self, batch_size: int, beta: float) -> \
            tp.Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self._waiting_for_feedback_frame_index != -1:
            raise Exception(f"Still waiting for the effect of frame {self._waiting_for_feedback_frame_index}")

        if batch_size > self.num_stored:
            raise Exception(f"Requested batch size {batch_size} > number of frames stored {self.num_stored}")

        if self.sampled_indices is not None:
            raise Exception(f"Previously sampled priorities have not been updated yet")

        obs_indices, next_obs_indices = self._sample_frame_indices(batch_size)

        obs_batch = np.stack([self._get_encoded_state_at(idx) for idx in obs_indices])
        act_batch = self._actions[obs_indices]
        rew_batch = self._rewards[obs_indices]
        next_obs_batch = np.stack([self._get_encoded_state_at(idx) for idx in next_obs_indices])
        done_mask = self._dones[obs_indices]
        weights = np.array([self._calculate_weight(i, beta) for i in obs_indices])
        self.sampled_indices = obs_indices
        return np2torch(obs_batch), np2torch(act_batch), np2torch(rew_batch), \
               np2torch(next_obs_batch), np2torch(done_mask), np2torch(weights)

    def update_priorities(self, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        # add a small number to ensure every transition can be sampled
        tempered_priorities = priorities + self.priority_eps
        for idx, priority in zip(self.sampled_indices, tempered_priorities):
            assert priority > 0
            assert 0 <= idx < self.num_stored

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
        self.sampled_indices = None
