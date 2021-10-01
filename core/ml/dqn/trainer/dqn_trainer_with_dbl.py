from typing import Tuple

import torch
from torch import Tensor

from core.ml.dqn.trainer.dqn_trainer import DqnTrainer


class DqnTrainerWithDbl(DqnTrainer):
    def _get_q_values(self, s_batch: Tensor, a_batch: Tensor, r_batch: Tensor, sp_batch: Tensor,
                      done_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get q_networks Q values and Target Q values
        Target_Qs(s) = r if done
                     = r + gamma * max_a'_q_network Q_target(s', a') # Double DQN

        :param s_batch: state batch (batch_size, n_channel, image_height, image_width)
        :param sp_batch: next state batch (batch_size, n_channel, image_height, image_width)
        :param a_batch: The action the agent took at each step (batch_size,)
        :param r_batch: The reward of each step (batch_size,)
        :param done_mask: whether the terminal state is reached (batch_size,)
        :return a tuple of q_network and target_network q_values
        """
        a_batch = a_batch.long().reshape(-1, 1)
        r_batch = r_batch.reshape(-1, 1)
        done_mask = done_mask.reshape(-1, 1)

        with torch.no_grad():
            # Double DQN: use the Q network to get the max indices and use target network to get the Q values
            indices = self.dqn.network(sp_batch).argmax(dim=1, keepdim=True)
            target_q_values = self.target_network(sp_batch).gather(1, indices)

        target_qs = r_batch + self.gamma * target_q_values * torch.logical_not(done_mask)

        # run a forward pass of q network and get the actions q values out
        qs = self.dqn.network(s_batch).gather(1, a_batch)
        return qs, target_qs
