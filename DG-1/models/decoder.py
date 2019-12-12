import torch
import torch.nn as nn
from utils import *
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),)

        self.seg1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.seg2 = nn.Sequential(

            nn.ConvTranspose2d(32, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32,16, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.seg3 = nn.Sequential(

            nn.ConvTranspose2d(16+32, 16, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 8, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )


        self.seg4 = nn.Sequential(
            nn.ConvTranspose2d(8+64, 3, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),

            nn.ConvTranspose2d(3, 3, 3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.Unflatten = Unflatten()

        self.fc = nn.Linear(300, 16384, bias=True).to(device)


    def forward(self, x):
        x = self.up(self.Unflatten(self.fc(x)))
        out1 = self.seg1(x).to(device)
        out2 = self.seg2(out1).to(device)
        out3 = self.seg3(torch.cat((out2,out1),dim=1)).to(device)
        out4 = self.seg4(torch.cat((out3,x),dim=1)).to(device)
        return out4
