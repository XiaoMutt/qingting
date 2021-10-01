from torch import nn

from core.ml.dqn.environment.environment import Environment
from core.ml.dqn.network.network import Network


class GoogleNatureNetwork(Network):
    def __init__(self, n_channels:int, image_height:int, image_width:int, action_space_dim:int):
        super(GoogleNatureNetwork, self).__init__(env)
        n_channels, image_height, image_width = env.frame_shape
        num_actions = env.action_space_dim
        self.network = nn.Sequential(
            nn.Conv2d(n_channels * env.frame_num_per_state, 32, kernel_size=(8, 8), stride=(4, 4),
                      padding=((4 - 1) * image_height - 4 + 8) // 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2),
                      padding=((2 - 1) * image_height - 2 + 4) // 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
                      padding=((1 - 1) * image_height - 1 + 3) // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(image_height * image_width * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_dim)
        )

        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight, gain=2 ** (1. / 2))
            if hasattr(m, 'bias'):
                nn.init.zeros_(m.bias)

        self.network.apply(init_weights)

    def forward(self, x):
        return self.network(x)
