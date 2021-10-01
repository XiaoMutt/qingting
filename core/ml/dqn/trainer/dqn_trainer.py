from logging import Logger
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from core.basis.cycler import Cycler
from core.basis.immutable import Immutable
from core.config import DEVICE
from core.ml.dqn.archiver.model_evaluator import ModelEvaluator
from core.ml.dqn.archiver.model_saver import ModelSaver
from core.ml.dqn.archiver.training_saver import TrainingSaver
from core.ml.dqn.environment.dqn_environment import DqnEnvironment
from core.ml.dqn.model.dqn import Dqn
from core.ml.dqn.replay_buffer.replay_buffer import ReplayBuffer
from core.ml.scheduler import LinearScheduler
from core.ml.tensorboard_writer import TensorboardWriter


class DqnTrainer(Immutable):
    def __init__(self,
                 logger: Logger,
                 env: DqnEnvironment,
                 dqn: Dqn,

                 gamma: float,
                 learning_start_step: int,
                 num_train_steps: int,
                 max_steps_per_episode: int,
                 batch_size: int,
                 grad_clip_value: float,

                 eps_scheduler: LinearScheduler,
                 lr_scheduler: LinearScheduler,
                 replay_buffer: ReplayBuffer,

                 learning_period: int,
                 target_update_period: int,
                 model_evaluation_period: int,
                 model_saving_period: int,
                 training_saving_period: int,
                 tensorboard_writing_period: int,

                 model_evaluator: ModelEvaluator,
                 model_saver: ModelSaver,
                 training_saver: TrainingSaver,
                 tensorboard_writer: TensorboardWriter
                 ):
        self.logger: Logger = logger
        self.env = env
        self.dqn = dqn

        self.gamma = gamma
        self.learning_start_step = learning_start_step
        self.num_train_steps = num_train_steps
        self.max_steps_per_episode = max_steps_per_episode
        self.batch_size = batch_size
        self.grad_clip_value = grad_clip_value

        self.eps_scheduler = eps_scheduler
        self.lr_scheduler = lr_scheduler
        self.replay_buffer: ReplayBuffer = replay_buffer

        self.learning_cycler = Cycler(learning_period)
        self.target_update_cycler = Cycler(target_update_period)
        self.model_evaluation_cycler = Cycler(model_evaluation_period)
        self.model_saving_cycler = Cycler(model_saving_period)
        self.training_saver_cycler = Cycler(training_saving_period)
        self.tensorboard_writing_cycler = Cycler(tensorboard_writing_period)

        self.model_evaluator: ModelEvaluator = model_evaluator
        self.model_saver: ModelSaver = model_saver
        self.training_saver: TrainingSaver = training_saver
        self.tensorboard_writer: TensorboardWriter = tensorboard_writer

        self.target_network = self.dqn.network_type(
            self.dqn.network.n_channels,
            self.dqn.network.image_height,
            self.dqn.network.image_width,
            self.dqn.network.action_space_dim
        )
        self.target_network.load_state_dict(self.dqn.network.state_dict())

        self.dqn.network.to(DEVICE)
        self.target_network.to(DEVICE)

        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.dqn.network.parameters())

    def train(self):
        """
        Perform Training for DQN
        """

        t = 0
        progress_bar = tqdm(total=self.num_train_steps, miniters=self.learning_cycler.period)
        while t < self.num_train_steps:
            frame = self.env.reset()

            with self.training_saver_cycler(t) as run, run:
                self.training_saver.start(t, self.env.image_state)

            for i in range(self.max_steps_per_episode):
                # add to replay buffer
                state = self.replay_buffer.add_frame(frame)
                # chose action according to current Q and exploration
                action, q_vals = self.get_action_and_q_vals_from_state(t, state)

                self.training_saver.add(i, frame, action, q_vals)

                # perform action in env
                # env moves to the next state
                frame, reward, done = self.env.step(action)

                # store the transition
                self.replay_buffer.add_feedback(action, reward, done)

                # store q values
                self.tensorboard_writer['BestQ'].append(max(q_vals))
                self.tensorboard_writer['Qs'].extend(q_vals)
                self.tensorboard_writer['Reward'].append(reward)

                if t < self.learning_start_step:
                    msg = f"Populating the memory {t}/{self.learning_start_step}..."
                    progress_bar.set_description(msg)
                else:
                    # train one step
                    with self.learning_cycler(t) as run, run:
                        loss, grad = self.train_step(t)
                        self.tensorboard_writer['Loss'].append(loss)
                        self.tensorboard_writer['Gradient'].append(grad)

                    # occasionally update target network with q network
                    with self.target_update_cycler(t) as run, run:
                        self.target_network.load_state_dict(self.dqn.network.state_dict())

                    # log training progress
                    with self.tensorboard_writing_cycler(t) as run, run:
                        values = self.tensorboard_writer.output(t)
                        msg = f'{t}: ' + ', '.join([f'{k}: {v:.2e}' for k, v in values.items()])
                        progress_bar.set_description(msg)

                    # occasionally save the weights
                    with self.model_saving_cycler(t) as run, run:
                        self.model_saver.save(t)

                t += 1
                progress_bar.update()

                if done or t >= self.num_train_steps:
                    break

            # one episode ends
            self.training_saver.save()

            # evaluate our policy
            with self.model_evaluation_cycler(t) as run, run:
                self.logger.info('evaluating...')
                rewards = self.model_evaluator.evaluate(t)
                average_reward = np.mean(rewards)
                sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

                msg = f"Average reward: {average_reward:04.2f} +/- {sigma_reward:04.2f}"
                self.logger.info(msg)
                self.tensorboard_writer['MeanEvalReward'].append(average_reward)
                self.tensorboard_writer.output(t)

        self.logger.info('Saving final model')
        self.model_saver.save(t)
        self.logger.info('Saved')

        self.logger.info('Evaluating final model')
        rewards = self.model_evaluator.evaluate(t)
        average_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        self.logger.info(f"Average reward: {average_reward:04.2f} +/- {sigma_reward:04.2f}")
        self.logger.info("### Training done. ###")

    def get_action_and_q_vals_from_state(self, t, state):
        # chose action according to current Q and exploration
        q_vals = self.dqn.get_q_vals(state)
        if np.random.rand() < self.eps_scheduler.value(t):
            action = np.random.randint(0, self.env.action_space_dim)
        else:
            action = np.argmax(q_vals)
        return action, q_vals

    def _get_q_values(self, s_batch: Tensor, a_batch: Tensor, r_batch: Tensor, sp_batch: Tensor,
                      done_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get q_networks Q values and Target Q values.
        Target_Qs(s) = r if done
                     = r + gamma * max_a'Q_target(s', a')

        Q network's gradients are recorded
        Target network's gradients are not recorded, because it's parameters are not not learned.

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
            target_q_values, _ = self.target_network(sp_batch).max(dim=1, keepdim=True)

        target_qs = r_batch + self.gamma * target_q_values * torch.logical_not(done_mask)

        # run a forward pass of q network and get the actions q values out
        qs = self.dqn.network(s_batch).gather(1, a_batch)
        return qs, target_qs

    def train_step(self, t: int) -> tuple:
        """
        Perform one training step
        :param t: time step
        :return: a tuple of loss and total loss norm
        """
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = self.replay_buffer.sample(self.batch_size)
        # reset Optimizer
        self.optimizer.zero_grad()

        # run calculate loss
        qs, target_qs = self._get_q_values(s_batch, a_batch, r_batch, sp_batch, done_mask_batch)
        loss = self.loss(qs, target_qs)

        # back prop loss
        loss.backward()

        # clip grad
        total_norm = torch.nn.utils.clip_grad_norm_(self.dqn.network.parameters(), self.grad_clip_value)

        # update parameters with optimizer
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr_scheduler.value(t)
        self.optimizer.step()
        return loss.item(), total_norm.item()
