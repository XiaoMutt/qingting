import numpy as np

from core.basis.immutable import Immutable
from core.ml.dqn.environment.image_state import ImageState


class EpisodeRecorder(Immutable):
    def __init__(self, image_state: ImageState):
        image, cursor = image_state.state
        self._initial_image = (image * 255).astype(np.uint8)  # convert to 8-bit
        self._initial_cursor = cursor.astype(np.uint8)  # convert to 8-bit
        self._actions = []
        self._q_vals = []

    def add(self, action, q_vals):
        self._actions.append(action)
        self._q_vals.append(q_vals)

    def save(self, path: str):
        actions = np.array(self._actions, dtype=np.uint8)
        q_vals = np.array(self._q_vals, dtype=np.float32)
        with open(path, 'wb') as writer:
            # initial_image
            tmp = self._initial_image.tobytes()
            writer.write(len(tmp).to_bytes(4, 'big'))
            writer.write(tmp)

            # initial_cursor
            tmp = self._initial_cursor.tobytes()
            writer.write(len(tmp).to_bytes(4, 'big'))
            writer.write(tmp)

            # actions
            tmp = actions.tobytes()
            writer.write(len(tmp).to_bytes(4, 'big'))
            writer.write(tmp)

            # q_vals
            tmp = q_vals.tobytes()
            writer.write(len(tmp).to_bytes(4, 'big'))
            writer.write(tmp)
