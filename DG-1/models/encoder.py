import torch.nn as nn
from utils import *
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.seg1 = nn.Sequential(
            nn.Conv2d(3,8, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(8,8, 3,padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        self.seg2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.seg3 = nn.Sequential(
            nn.Conv2d(16+8, 32, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.seg4 = nn.Sequential(
            nn.Conv2d(32+3, 64, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2),
        )
        self.Flatten = Flatten()

        self.fc = nn.Linear(16384, 300, bias=True).to(device)

    def forward(self, x):

        out1 = self.seg1(x).to(device)
        out2 = self.seg2(out1).to(device)
        out3 = self.seg3(torch.cat((out2,out1),dim=1)).to(device)
        out4 = self.seg4(torch.cat((out3,x),dim=1)).to(device)
        out = self.fc(self.Flatten(out4)).to(device)
        return out
