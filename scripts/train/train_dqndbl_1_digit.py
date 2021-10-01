import os

from core.ml.dqn.archiver.model_evaluator import ModelEvaluator
from core.ml.dqn.archiver.model_saver import ModelSaver
from core.ml.dqn.archiver.training_saver import TrainingSaver
from core.ml.dqn.environment.dqn_environment import DqnEnvironment
from core.ml.dqn.image_composer import DqnImageComposer
from core.ml.dqn.model.dqn import Dqn
from core.ml.dqn.network.res_network import ResNetwork
from core.ml.dqn.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from core.ml.dqn.trainer.dqn_trainer_with_dbl import DqnTrainerWithDbl
from core.ml.scheduler import LinearScheduler
from core.ml.tensorboard_writer import TensorboardWriter
from scripts.config import TRAINING, PROJECT_FOLDER
from scripts.train.config import train_background_image_handler, train_digit_label_handler, train_digit_image_handler

if __name__ == '__main__':
    training = TRAINING()

    env = DqnEnvironment(
        DqnImageComposer(
            train_background_image_handler,
            train_digit_image_handler,
            train_digit_label_handler),
        min_num_of_digits=1,
        max_num_of_digits=1,
        frame_num_per_state=4
    )

    # dqn = Dqn(env, config, ResNetwork)
    dqn = Dqn(logger=training.LOGGER, frame_shape=env.frame_shape, frame_mum_per_state=env.frame_num_per_state,
              action_space_dim=env.action_space_dim, network_type=ResNetwork, epsilon=0.01,
              model_load_path=os.path.join(PROJECT_FOLDER, 'data', 'dqn-051819.weights'))

    trainer = DqnTrainerWithDbl(
        logger=training.LOGGER,
        env=env,
        dqn=dqn,

        gamma=0.99,
        learning_start_step=50_000,
        num_train_steps=12_000_000,
        max_steps_per_episode=500,
        batch_size=32,
        grad_clip_value=10,

        eps_scheduler=LinearScheduler(0.5, 0.05, 4_000_000),
        lr_scheduler=LinearScheduler(1e-4, 5e-5, 500_000),
        replay_buffer=UniformReplayBuffer(size=500_000,
                                          frame_num_per_state=env.frame_num_per_state,
                                          frame_shape=env.frame_shape),

        learning_period=5,
        target_update_period=10_000,
        model_evaluation_period=500_000,
        model_saving_period=500_000,
        training_saving_period=100_000,
        tensorboard_writing_period=50,

        model_evaluator=ModelEvaluator(output_folder=training.RECORD_SAVE_FOLDER, env=env, dqn=dqn,
                                       num_evaluation_episodes=20, maximum_num_steps=500),
        model_saver=ModelSaver(output_folder=training.MODEL_SAVE_FOLDER, dqn=dqn),
        training_saver=TrainingSaver(output_folder=training.RECORD_SAVE_FOLDER),
        tensorboard_writer=TensorboardWriter(output_folder=training.OUTPUT_FOLDER)
    )
    trainer.train()
