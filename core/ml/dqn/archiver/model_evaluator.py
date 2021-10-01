import os

from tqdm import tqdm

from core.ml.dqn.archiver.archiver import Archiver
from core.ml.dqn.environment.dqn_environment import DqnEnvironment
from core.ml.dqn.model.dqn import Dqn
from core.ml.dqn.recorder.episode_player import EpisodePlayer
from core.ml.dqn.recorder.episode_recorder import EpisodeRecorder


class ModelEvaluator(Archiver):
    def __init__(self, output_folder: str,
                 env: DqnEnvironment, dqn: Dqn,
                 num_evaluation_episodes: int, maximum_num_steps: int = 500,
                 render: bool = False):
        assert 0 < num_evaluation_episodes < 100, \
            f"0< num_evaluation_episodes must < 100, but {num_evaluation_episodes} is given"
        super(ModelEvaluator, self).__init__(output_folder, num_evaluation_episodes * 5)
        self.env = env
        self.dqn = dqn
        self.num_evaluation_episodes = num_evaluation_episodes
        self.maximum_num_steps = maximum_num_steps
        self.episode_player = EpisodePlayer() if render else None

    def _render(self, t, frame, action, q_vals):
        if self.episode_player:
            self.episode_player.render(t, frame, action, q_vals)

    def evaluate(self, t: int) -> list:
        """
        Evaluate and return
        :param t:
        :return: a list of rewards or None if evaluate does not run
        """
        # replay memory to play
        rewards = []
        for i in tqdm(range(self.num_evaluation_episodes)):
            total_reward = 0

            frame = self.env.reset()
            record = EpisodeRecorder(self.env.image_state)
            for j in range(0, self.maximum_num_steps):
                action, q_vals = self.dqn.act_to(self.env.image_state)

                self._render(j, frame, action, q_vals)
                record.add(action, q_vals)

                # perform action in env
                frame, reward, done = self.env.step(action)

                # store in replay memory
                self.dqn.add_feedback(action, reward, done)

                # count reward
                total_reward += reward

                if done:
                    break

            rewards.append(total_reward)
            # save the frames in file
            path = os.path.join(self.output_folder, f'evaluation_at_t{t:09d}-{i:02d}.record')
            record.save(path)
            self._register_archived(path)
        return rewards
