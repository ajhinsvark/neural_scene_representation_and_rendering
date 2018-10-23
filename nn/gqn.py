from .dqn import DQN
from .vae import VAEDecoder

import torch
from torch import nn

class GQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.dqn = DQN()
        self.gen = VAEDecoder()

    def forward(self, context_views, context_frames, target_view):
        representation = self.dqn(context_views, context_frames)
        return self.gen(representation, target_view)
        