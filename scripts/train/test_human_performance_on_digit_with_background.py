import matplotlib.pyplot as plt

from core.ml.isn.image_composer import InnImageComposer
from scripts.train.config import dev_background_image_handler, dev_digit_image_handler, dev_digit_label_handler

if __name__ == '__main__':
    dev_image_composer = InnImageComposer(
        background_image_handler=dev_background_image_handler,
        digit_image_handler=dev_digit_image_handler,
        digit_label_handler=dev_digit_label_handler
    )

    total = 100
    count = 0
    data_batch, label_batch, pgrid_batch = dev_image_composer.compose(total)
    plt.ion()
    plt.show()
    for i in range(total):
        img = data_batch[i][0]
        label = label_batch[i]
        plt.matshow(img, 0)
        plt.pause(0.001)
        human_input = input('=?')
        if human_input == '':
            human_input = '10'
        correct = (int(human_input) == int(label))
        if correct:
            count += 1
            print("Correct")
        else:
            print(f'was {label}')

    print(count / total, count)
