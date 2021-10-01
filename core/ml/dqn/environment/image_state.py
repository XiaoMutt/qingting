import numpy as np

from core.basis.protected import Protected


class ImageState(Protected):
    DIRECTION_MASKS = np.array([
        np.array([0, -1]),  # left
        np.array([-1, 0]),  # up
        np.array([0, 1]),  # right
        np.array([1, 0]),  # down
        np.array([-1, -1]),  # up, left
        np.array([-1, 1]),  # up,right
        np.array([1, 1]),  # down, right
        np.array([1, -1])  # down, left
    ], dtype=np.int32)

    ACTION_NAMES = (
        "▣",
        "✓",
        "←",
        "↑",
        "→",
        "↓",
        "↖",
        "↗",
        "↘",
        "↙",
    )

    NUM_ACTIONS = len(ACTION_NAMES)

    def __init__(self, image: np.ndarray, cursor: np.ndarray):
        """
        ImageState handles
        - the movement of the cursor on the image
        - the end action which hides the cursor
        - the strike action which paint the cursor area to 0.5
        - the creation of the frame (for replay buffer) given the current image and cursor
        The state of the ImageState is solely determined by the image and the cursor
        :param image: the image ndarray
        :param cursor: a 2-d ndarray representing the cursor box:
            - [[leftTopX, leftTopY], [rightBottomX, rightBottomY]]
            - leftTopX and Y are inclusive
            - rightBottomX and Y are exclusive
        """
        self._image = image
        self._cursor = cursor
        self._image_shape = np.array(image.shape, np.int32)

    @property
    def image(self) -> np.ndarray:
        return np.copy(self._image)

    @property
    def cursor(self) -> np.ndarray:
        return np.copy(self._cursor)

    @property
    def state(self):
        return self.image, self.cursor

    @property
    def frame(self):
        res = np.copy(self._image)
        a, b = self._cursor
        res[a[0], a[1]:b[1]] = 1
        res[b[0] - 1, a[1]:b[1]] = 1
        res[a[0]:b[0], a[1]] = 1
        res[a[0]:b[0], b[1] - 1] = 1
        return res

    @property
    def done(self):
        return self._cursor.sum() == 0

    def act(self, action: int):
        if action == 0:
            self._cursor[:] = 0
        elif action == 1:
            # strike
            a, b = self._cursor
            self._image[a[0]:b[0], a[1]:b[1]] = 0.5
        else:
            mask = self.DIRECTION_MASKS[action - 2]
            tmp = self._cursor + mask
            # keep valid direction moves
            self._cursor += ((tmp[0] >= 0) & (tmp[1] <= self._image_shape)) * mask
