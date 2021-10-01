from core.basis.immutable import Immutable


class DqnConfig(Immutable):
    def __init__(self,
                 replay_buffer_size,  # the size of the ReplayBuffer
                 batch_size,  # How many samples to take from ReplayBuffer each time

                 num_train_steps,
                 num_episodes_test,
                 learning_start_step,

                 log_frequency,
                 learning_frequency,
                 evaluation_frequency,
                 record_frequency,
                 target_update_frequency,
                 saving_frequency,

                 soft_epsilon,  # epsilon used during evaluation
                 gamma,
                 grad_clip_value,

                 eps_scheduler,
                 lr_scheduler,

                 stopwatch,
                 logger,
                 output_folder,
                 record_save_folder,
                 model_load_path,
                 model_save_folder,
                 ):
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.num_train_steps = num_train_steps
        self.num_episodes_test = num_episodes_test
        self.learning_start_step = learning_start_step
        self.log_frequency = log_frequency
        self.learning_frequency = learning_frequency
        self.evaluation_frequency = evaluation_frequency
        self.record_frequency = record_frequency
        self.target_update_frequency = target_update_frequency
        self.saving_frequency = saving_frequency

        self.soft_epsilon = soft_epsilon
        self.gamma = gamma
        self.grad_clip_value = grad_clip_value

        self.eps_scheduler = eps_scheduler
        self.lr_scheduler = lr_scheduler

        self.stopwatch = stopwatch
        self.logger = logger
        self.output_folder = output_folder
        self.record_save_folder = record_save_folder
        self.model_load_path = model_load_path
        self.model_save_folder = model_save_folder
