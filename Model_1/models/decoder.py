import torch.nn as nn
from lib.config import *

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(300,512,bias=True).to(device),
            nn.ReLU(),
            nn.Linear(512, 28 * 28, bias=True).to(device),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, x):
        out = self.decoder(x).to(device)
        return out
