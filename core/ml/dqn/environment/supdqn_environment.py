import numpy as np

from core.ml.dqn.environment.dqn_environment import DqnEnvironment


class SupDqnEnvironment(DqnEnvironment):
    def _find_closet(self):
        destination = None
        min_dist = 4294967295
        cursor = self.image_state.cursor + self.cursor_padding
        for idx in self._available_region_indices:
            dest = self._regions[idx]
            tmp = np.max(np.abs(dest[0] - cursor[0]))
            if min_dist > tmp:
                destination = dest
                min_dist = tmp

        return destination, min_dist

    def get_optimal_action(self):
        # greedy method: pick the closest destination first
        # for 2 destinations, it is the optimal
        if self._image_state.done:
            raise Exception(f"Episode already done! No action available")

        destination, min_dist = self._find_closet()
        if destination is None:
            # stop
            return 0

        if min_dist < 3:  # 2*sqrt(2)
            # strike at high probability
            if np.random.random() < 0.95:
                return 1

        move = np.sign(destination[0] - self.image_state.cursor[0] - self.cursor_padding)

        if np.all(move == 0):
            # perfectly centered
            return 1
        for idx, m in enumerate(self._image_state.DIRECTION_MASKS):
            if np.all(move == m):
                return idx + 2

        raise Exception("Cannot find move: this should not occur.")
