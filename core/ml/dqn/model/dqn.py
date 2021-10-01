import os
import typing as tp
from logging import Logger
from typing import Type

import numpy as np
import torch
from torch import Tensor

from core.basis.immutable import Immutable
from core.ml.dqn.environment.image_state import ImageState
from core.ml.dqn.network.network import Network
from core.ml.dqn.replay_buffer.replay_buffer import ReplayBuffer


class Dqn(Immutable):
    def __init__(self,
                 logger: Logger,
                 frame_shape: tuple,
                 frame_mum_per_state: int,
                 action_space_dim: int,
                 network_type: Type[Network],
                 epsilon: float,
                 model_load_path: tp.Union[str, None] = None):
        """
        Dqn: Deep Q network. It handles
        :param logger:
        :param frame_shape:
        :param frame_mum_per_state:
        :param action_space_dim:
        :param network_type:
        :param epsilon:
        :param model_load_path:
        """
        self.logger = logger
        self.network_type = network_type
        self.epsilon = epsilon

        self.action_space_dim = action_space_dim
        self.replay_buffer = ReplayBuffer(1024, frame_mum_per_state, frame_shape)

        frame_channels, frame_height, frame_width = frame_shape
        self.network = network_type(frame_mum_per_state * frame_channels, frame_height, frame_width, action_space_dim)

        if model_load_path:
            if os.path.isfile(model_load_path):
                self.logger.info(f'Loading parameters from file: {model_load_path}')
                self.network.load_state_dict(torch.load(model_load_path, map_location='cpu'))
                self.logger.info('Load successful!')
            else:
                self.logger.info(f'load_path {model_load_path} is invalid')
                self.logger.info('Parameters initialized randomly')
        else:
            self.logger.info('Parameters initialized randomly')

    def get_q_vals(self, state: Tensor) -> list:
        """
        Return best action
        Returns: a list of Q(state, action)
        """
        with torch.no_grad():
            q_vals = self.network(state).squeeze().cpu()
        res = q_vals.tolist()
        return res

    def act_to(self, image_state: ImageState) -> tuple:
        """
        Returns action with a epsilon strategy
        """
        state = self.replay_buffer.add_frame(image_state.frame)
        q_vals = self.get_q_vals(state)
        if np.random.random() < self.epsilon:
            res = np.random.randint(0, self.action_space_dim), q_vals
        else:
            res = np.argmax(q_vals).squeeze(), q_vals
        return res

    def add_feedback(self, action: int, reward: float, done: bool) -> None:
        self.replay_buffer.add_feedback(action, reward, done)

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)
