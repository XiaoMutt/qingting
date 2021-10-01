from core.basis.logger import setup_logger
from core.ml.isn.config import IsnConfig
from core.ml.isn.image_composer import InnImageComposer
from core.ml.isn.isn import ISN
from core.ml.isn.resnet import PrbCubeResNet
from core.ml.isn.trainer import IsnTrainer
from core.ml.scheduler import LinearScheduler
from scripts.config import TRAINING

from scripts.train.config import train_digit_image_handler, train_background_image_handler, train_digit_label_handler, \
    dev_digit_label_handler, dev_digit_image_handler, dev_background_image_handler

if __name__ == '__main__':
    training = TRAINING()
    train_image_composer = InnImageComposer(
        background_image_handler=train_background_image_handler,
        digit_image_handler=train_digit_image_handler,
        digit_label_handler=train_digit_label_handler
    )

    dev_image_composer = InnImageComposer(
        background_image_handler=dev_background_image_handler,
        digit_image_handler=dev_digit_image_handler,
        digit_label_handler=dev_digit_label_handler
    )

    config = IsnConfig(
        train_batch_size=32,
        num_train_steps=1_000_000,
        test_batch_size=128,

        log_frequency=50,
        eval_frequency=10_000,
        saving_frequency=100_000,

        grad_clip_value=10,
        lr_scheduler=LinearScheduler(1e-3, 1e-5, 500_000),

        logger=setup_logger(training.LOG_SAVE_FOLDER, 'isn'),
        output_folder=training.OUTPUT_FOLDER,
        model_load_path=None,
        model_save_folder=training.MODEL_SAVE_FOLDER
    )

    isn = ISN(config, PrbCubeResNet)
    trainer = IsnTrainer(train_image_composer=train_image_composer,
                         dev_image_composer=dev_image_composer,
                         config=config,
                         isn=isn)
    trainer.train()
