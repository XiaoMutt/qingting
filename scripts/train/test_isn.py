import matplotlib.pyplot as plt
import torch

from core.basis.logger import setup_logger
from core.ml.isn.config import IsnConfig
from core.ml.isn.image_composer import InnImageComposer
from core.ml.isn.isn import ISN
from core.ml.isn.resnet import PrbCubeResNet
from core.ml.isn.trainer import IsnTrainer
from core.ml.scheduler import LinearScheduler
from scripts.config import TRAINING
from scripts.train.config import train_digit_image_handler, train_background_image_handler, train_digit_label_handler, \
    test_background_image_handler, test_digit_image_handler, test_digit_label_handler

if __name__ == '__main__':
    train_image_composer = InnImageComposer(
        background_image_handler=train_background_image_handler,
        digit_image_handler=train_digit_image_handler,
        digit_label_handler=train_digit_label_handler
    )

    test_image_composer = InnImageComposer(
        background_image_handler=test_background_image_handler,
        digit_image_handler=test_digit_image_handler,
        digit_label_handler=test_digit_label_handler
    )
    training = TRAINING()
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
        model_load_path=r'/Volumes/XiaoSSD/runs/isn/outputs/1622626340/models/dqn-1000000.weights',
        model_save_folder=training.MODEL_SAVE_FOLDER
    )

    isn = ISN(config, PrbCubeResNet)
    trainer = IsnTrainer(train_image_composer=train_image_composer,
                         dev_image_composer=test_image_composer,
                         config=config,
                         isn=isn)
    data_batch, label_batch, pgrid_batch = trainer.dev_ic.compose(1)
    label, pgrid = trainer.isn.predict(data_batch)

    correct = (label == label_batch)
    label_accuracy = correct.sum() / label.numel()

    correct_digits = torch.logical_and(correct, label != 10)
    match = (pgrid == pgrid_batch)[correct_digits]
    if match.numel() > 0:
        pgrid_accuracy = match.sum() / match.numel()
    else:
        pgrid_accuracy = 1

    print(label_accuracy, pgrid_accuracy)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(4, 1.8), sharex=True, sharey=True)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
    ax0.axes.xaxis.set_visible(False)
    ax0.axes.yaxis.set_visible(False)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    ax0.matshow(data_batch[0][0].numpy())
    ax0.set_title(f'Input Region {label_batch[0]}', fontsize=10)
    ax1.matshow(pgrid[0][0].numpy())
    ax1.set_title(f'ISN Prediction {label[0]}', fontsize=10)
    ax2.matshow(pgrid_batch[0][0].numpy())
    ax2.set_title(f'Ground Truth {label_batch[0]}', fontsize=10)
    plt.show()
