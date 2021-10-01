from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from core.ml.dqn.model.dbl_dqn import DblDqn


class PerDblDqn(DblDqn):

    def _calc_loss(self, s_batch: Tensor, a_batch: Tensor, r_batch: Tensor, sp_batch: Tensor,
                   done_mask: Tensor) -> Tensor:
        """
        Calculate the Huber loss (SmoothL1Loss) of this step.
                    loss = SmoothL1Loss(Q_sample(s) - Q(s, a))
        :param s_batch: state batch (batch_size, n_channel, image_height, image_width)
        :param sp_batch: next state batch (batch_size, n_channel, image_height, image_width)
        :param a_batch: The action the agent took at each step (batch_size,)
        :param r_batch: The reward of each step (batch_size,)
        :param done_mask: whether the terminal state is reached (batch_size,)
        :return a Tensor contains the loss for each batch sample (batch_size,)
        """

        qs, target_qs = self._get_q_values(s_batch, a_batch, r_batch, sp_batch, done_mask)
        loss = nn.functional.smooth_l1_loss(qs, target_qs, reduction="none")
        return loss

    def update_step(self, lr: float, sampled_batch: tuple) -> Tuple[float, float, np.ndarray]:
        """
        Performs an update of parameters by sampling from replay_buffer
        :param lr: learning rate
        :param sampled_batch: a tuple contains:
        s_batch: state batch
        a_batch: action batch
        r_batch: reward batch
        sp_batch: state prime batch (next state batch)
        done_mask_batch: done mask batch
        weights: importance sampling weights for adjust loss
        :return: tuple of loss and total loss norm
        """
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch, weights = sampled_batch
        # reset Optimizer
        self.optimizer.zero_grad()

        # run calculate loss
        element_wise_loss = self._calc_loss(s_batch, a_batch, r_batch, sp_batch, done_mask_batch)
        loss = torch.mean(element_wise_loss * weights)

        # back prop loss
        loss.backward()

        # clip grad
        total_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_value)

        # update parameters with optimizer
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        return loss.item(), total_norm.item(), element_wise_loss.detach().cpu().numpy()
