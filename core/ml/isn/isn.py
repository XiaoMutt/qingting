import os
from typing import Tuple

import torch
from torch import Tensor


class ISN(object):
    def __init__(self, config, network):
        self.config = config
        self.logger = self.config.logger
        self.network = network()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if self.config.model_load_path:
            if os.path.isfile(self.config.model_load_path):
                self.logger.info(f'Loading parameters from file: {self.config.model_load_path}')
                self.network.load_state_dict(torch.load(self.config.model_load_path, map_location='cpu'))
                self.logger.info('Load successful!')
            else:
                self.logger.info(f'load_path {self.config.model_load_path} is invalid')
                self.logger.info('Parameters initialized randomly')
        else:
            self.logger.info('Parameters initialized randomly')

        self.optimizer = torch.optim.Adam(self.network.parameters())

    def predict(self, data_batch) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            label, pgrid = self.network(data_batch)
        label_ = label.argmax(dim=1)

        pgrid_ = torch.ones_like(pgrid)
        pgrid_[pgrid <= 0] = -1

        return label_, pgrid_

    def _calc_loss(self, label_predicted, pgrid_predicted, label_batch, pgrid_batch):
        label_cost = self.cross_entropy(label_predicted, label_batch)
        label_ = label_predicted.argmax(dim=1)

        incorrectly_predicted_mask = torch.ne(label_, label_batch)
        correctly_predicted_digit_mask = torch.logical_and(torch.eq(label_, label_batch), torch.ne(label_, 10))

        pgrid_cost = torch.clamp(1 - pgrid_batch * pgrid_predicted, min=0)
        pgrid_cost = pgrid_cost.mean((1, 2, 3))
        pgrid_cost = correctly_predicted_digit_mask * pgrid_cost + incorrectly_predicted_mask * 2
        pgrid_cost = pgrid_cost.mean()
        assert label_cost >= 0 and pgrid_cost >= 0
        loss = label_cost + pgrid_cost
        return loss

    def update(self, lr, data_batch, label_batch, pgrid_batch):
        # reset Optimizer
        self.optimizer.zero_grad()

        # forward
        pred_label, pred_pgrid = self.network(data_batch)
        # run calculate loss
        loss = self._calc_loss(pred_label, pred_pgrid, label_batch, pgrid_batch)
        # back prop loss
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_value)
        # update parameters with optimizer
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        return loss.item(), total_norm.item()
