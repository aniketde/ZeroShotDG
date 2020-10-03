import torch.nn as nn
from lib.config import *
from lib.utils import Unflatten
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(300, 4096,bias =True).to(device),
            nn.ReLU(),

            Unflatten(),

            nn.ConvTranspose2d(256,128, 3,padding=1,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128,128, 3,padding=1,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.ConvTranspose2d(128,64, 3,padding=1,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64,64, 3,padding=1,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.ConvTranspose2d(64,32, 3,padding=1,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32,32, 3,padding=1,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.ConvTranspose2d(32,3, 3,padding=1,stride=1),
            nn.ReLU(),
        ).to(device)

    def forward(self, x):
        out = self.decoder(x).to(device)
        return out