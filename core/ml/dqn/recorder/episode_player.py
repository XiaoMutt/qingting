import matplotlib.pyplot as plt
import numpy as np

from core.ml.dqn.environment.image_state import ImageState


class EpisodePlayer(object):
    def __init__(self, interval=0.01):
        fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [0.85, 0.15]})
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        ax[1].axes.xaxis.set_visible(False)
        self._render_ax = ax
        self._interval = interval

    def play(self, record_file_path):
        with open(record_file_path, 'rb') as reader:
            # initial image
            tmp = reader.read(4)
            size = int.from_bytes(tmp, 'big')
            tmp = reader.read(size)
            initial_image = np.frombuffer(tmp, dtype=np.uint8) / 255  # convert back to float
            initial_image = initial_image.reshape((int(initial_image.size ** 0.5), -1))

            # initial cursor
            tmp = reader.read(4)
            size = int.from_bytes(tmp, 'big')
            tmp = reader.read(size)
            initial_cursor = np.frombuffer(tmp, dtype=np.uint8).astype(np.int32)
            initial_cursor = initial_cursor.reshape((int(initial_cursor.size ** 0.5), -1))

            # actions
            tmp = reader.read(4)
            size = int.from_bytes(tmp, 'big')
            tmp = reader.read(size)
            actions = np.frombuffer(tmp, dtype=np.uint8).reshape((-1,))

            # q_vals
            tmp = reader.read(4)
            size = int.from_bytes(tmp, 'big')
            tmp = reader.read(size)
            q_vals = np.frombuffer(tmp, dtype=np.float32).reshape((-1, ImageState.NUM_ACTIONS))

        image_state = ImageState(initial_image, initial_cursor)
        for t in range(len(actions)):
            frame = image_state.frame
            self.render(t, frame, actions[t], q_vals[t])
            image_state.act(actions[t])

    def render(self, t: int, frame: np.ndarray, action: int, q_vals: np.ndarray):
        q_vals = np.array(q_vals).reshape((10, 1))
        action_name = ImageState.ACTION_NAMES[action]
        self._render_ax[0].cla()
        self._render_ax[0].matshow(frame)
        self._render_ax[0].set_title(f"frame {t}: action {action_name}")
        self._render_ax[1].cla()
        self._render_ax[1].matshow(q_vals)
        self._render_ax[1].set_aspect(0.4)
        self._render_ax[1].set_yticks(np.arange(10))
        self._render_ax[1].set_yticklabels(ImageState.ACTION_NAMES)
        self._render_ax[1].axes.yaxis.get_ticklabels()[action].set_color('red')
        self._render_ax[1].set_title("Q values")
        max_q_action = np.argmax(q_vals)
        for i, q_val in enumerate(q_vals.squeeze()):
            self._render_ax[1].text(0, i, f"{q_val:.4f}", ha="center", va="center",
                                    color="r" if i == max_q_action else "w")
        plt.pause(self._interval)
