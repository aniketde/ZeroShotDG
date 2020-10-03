import torch.nn as nn
from lib.config import *
from lib.utils import Flatten

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,32, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),

            nn.Conv2d(32,64, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64,64, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2),

            nn.Conv2d(64,128, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128,128, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128,256, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256,256, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            Flatten(),

            nn.Linear(12544, 300, bias=True).to(device),
        ).to(device)

    def forward(self, x):
        out = self.encoder(x).to(device)

        return out