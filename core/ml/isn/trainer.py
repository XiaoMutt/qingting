import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.ml.isn.config import IsnConfig
from core.ml.isn.image_composer import InnImageComposer
from core.ml.isn.isn import ISN


class IsnTrainer(object):
    def __init__(self, train_image_composer: InnImageComposer,
                 dev_image_composer: InnImageComposer,
                 config: IsnConfig, isn: ISN):
        self.config = config
        self.train_ic = train_image_composer
        self.dev_ic = dev_image_composer
        self.isn = isn

        self.logger = self.config.logger
        self.summary_writer = SummaryWriter(self.config.output_folder, max_queue=100000)

    def train(self):
        """
        Perform Training for DQN
        """

        pbar = tqdm(total=self.config.num_train_steps)

        for t in range(self.config.num_train_steps):
            loss, grad_norm = self.train_step(t)
            # log training progress
            if t % self.config.log_frequency == 0:
                self.summary_writer.add_scalar('loss', loss, t)
                self.summary_writer.add_scalar('grad_norm', grad_norm, t)
                pbar.set_description(f'loss: {loss:.2e} grad_norm: {grad_norm:.2e}')

            if t % self.config.eval_frequency == 0:
                label_accuracy, pgrid_accuracy = self.evaluate(t)
                self.summary_writer.add_scalar('eval_label_accuracy', label_accuracy, t)
                self.summary_writer.add_scalar('eval_pgrid_accuracy', pgrid_accuracy, t)

            if t % self.config.saving_frequency == 0:
                self.save_model(t)
            pbar.update()

        # last words
        self.logger.info('Saving model')
        self.save_model(self.config.num_train_steps)
        self.logger.info("### Training done. ###")

    def train_step(self, t: int):
        """
        Perform one training step
        :param t: time step
        :return:
        """
        data_batch, label_batch, pgrid_batch = self.train_ic.compose(self.config.train_batch_size)
        loss, norm = self.isn.update(self.config.lr_scheduler.value(t), data_batch, label_batch, pgrid_batch)
        return loss, norm

    def evaluate(self, t: int):
        """
        Evaluation with same procedure as the training
        """

        def calc():
            label, pgrid = self.isn.predict(data_batch)
            correct = (label == label_batch)
            label_accuracy = correct.sum() / label.numel()

            correct_digits = torch.logical_and(correct, label != 10)
            match = (pgrid == pgrid_batch)[correct_digits]
            if match.numel() > 0:
                pgrid_accuracy = match.sum() / match.numel()
            else:
                pgrid_accuracy = 1
            return label_accuracy, pgrid_accuracy

        self.logger.info("Evaluating...")
        data_batch, label_batch, pgrid_batch = self.train_ic.compose(self.config.train_batch_size)
        a, b = calc()
        self.logger.info(
            f"Evaluation on train set at iteration: {t}: "
            f"label accuracy: {a}, pgrid_accuracy: {b}")

        data_batch, label_batch, pgrid_batch = self.dev_ic.compose(self.config.train_batch_size)
        a, b = calc()
        self.logger.info(
            f"Evaluation on dev set at iteration: {t}: "
            f"label accuracy: {a}, pgrid_accuracy: {b}")

        return a, b

    def save_model(self, t):
        """
        Saves network weights
        """
        # remove old ones
        for exist in os.listdir(self.config.model_save_folder):
            path = os.path.join(self.config.model_save_folder, exist)
            os.remove(path)

        # save new ones
        torch.save(self.isn.network.state_dict(),
                   os.path.join(self.config.model_save_folder, f'isn-{t:06d}.weights'))
