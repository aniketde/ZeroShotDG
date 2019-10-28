import torch.nn as nn
from lib.config import *


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512, bias=True).to(device),
            nn.ReLU(),
            nn.Linear(512, 300, bias=True).to(device),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, x):
        out = self.encoder(x).to(device)

        return out
