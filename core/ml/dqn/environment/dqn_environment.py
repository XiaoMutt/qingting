import numpy as np

from core.basis.utils import iou
from core.ml.dqn.environment.image_state import ImageState
from core.ml.dqn.image_composer import DqnImageComposer


class DqnEnvironment(object):
    def __init__(self, image_composer: DqnImageComposer,
                 min_num_of_digits: int, max_num_of_digits: int,
                 frame_num_per_state: int, iou_threshold: float = None):

        self._regions = None
        self._segments = None
        self._available_region_indices = None
        self._num_chance_remains = None
        self._image_state = None

        self._image_composer = image_composer
        self._min_num_of_digits = min_num_of_digits
        self._max_num_of_digits = max_num_of_digits
        self._moving_cost = -0.01
        self._wrong_strike_cost = -0.1

        self.action_space_dim = 10  # 1 end + 1 strike + number of directions
        self.frame_shape = (1, *image_composer.image_size)
        self.frame_num_per_state = frame_num_per_state
        self.image_size = image_composer.image_size
        self.cursor_padding = 2  # the padding to enlarge the cursor based on the digit size
        self.cursor_size = image_composer.digit_size + 2 * self.cursor_padding

        self._iou_threshold = np.prod(image_composer.digit_size - 2) / \
                              (np.prod(image_composer.digit_size) + np.prod(self.cursor_size)
                               - np.prod(image_composer.digit_size - 2)) if iou_threshold is None else iou_threshold

    def _get_strike_reward(self):
        best_score = self._iou_threshold
        best_idx = None
        for idx in self._available_region_indices:
            region = self._regions[idx]
            score = iou(self._image_state.cursor, region)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            self._available_region_indices.remove(best_idx)
            return 1
        else:
            return self._wrong_strike_cost

    def reset(self):
        """
        Reset environment and return a Tensor represent a frame
        :return:
        """
        num_of_digits = np.random.randint(self._min_num_of_digits, self._max_num_of_digits + 1)
        image, regions, segments = self._image_composer.compose(num_of_digits)
        top_left = np.random.randint([0, 0], self.image_size - self.cursor_size)
        bottom_right = top_left + self.cursor_size
        cursor = np.array([top_left, bottom_right], dtype=np.int32)
        self._image_state = ImageState(image, cursor)

        self._regions = regions
        self._segments = segments
        self._available_region_indices = set(range(len(regions)))
        self._num_chance_remains = len(regions)

        return self._image_state.frame

    def step(self, action: int) -> tuple:
        """
        Perform action and return state, reward, done
        :param action: the action
        :return: a tuple of state, reward, done
        """

        if self._image_state.done:
            raise Exception("Episode already done!")

        self._image_state.act(action)
        if action == 0:
            # end
            reward = -len(self._available_region_indices)
        elif action == 1:
            # strike
            reward = self._get_strike_reward()
        else:
            reward = self._moving_cost

        return self._image_state.frame, reward, self._image_state.done

    @property
    def image_state(self) -> ImageState:
        """
        Return a copy of ImageState
        :return:
        """
        return ImageState(*self._image_state.state)

    def render(self):
        pass

    def get_optimal_action(self):
        raise NotImplementedError
