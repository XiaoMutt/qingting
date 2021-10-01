from typing import Optional

from core.basis.immutable import Immutable
from core.ml.scheduler import LinearScheduler


class IsnConfig(Immutable):
    def __init__(self,
                 train_batch_size: int,
                 num_train_steps: int,
                 test_batch_size: int,

                 log_frequency: int,
                 eval_frequency: int,
                 saving_frequency: int,

                 grad_clip_value: float,
                 lr_scheduler: LinearScheduler,

                 logger,

                 output_folder: str,
                 model_save_folder: str,
                 model_load_path: Optional[str]
                 ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_train_steps = num_train_steps

        self.log_frequency = log_frequency
        self.eval_frequency = eval_frequency
        self.saving_frequency = saving_frequency

        self.grad_clip_value = grad_clip_value
        self.lr_scheduler = lr_scheduler

        self.logger = logger
        self.output_folder = output_folder
        self.model_save_folder = model_save_folder
        self.model_load_path = model_load_path
