from torch import nn


class Network(nn.Module):
    def __init__(self, n_channels: int, image_height: int, image_width: int, action_space_dim: int):
        super(Network, self).__init__()
        self.n_channels = n_channels
        self.image_height = image_height
        self.image_width = image_width
        self.action_space_dim = action_space_dim
