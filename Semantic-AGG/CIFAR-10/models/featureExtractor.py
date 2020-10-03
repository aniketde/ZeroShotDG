import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from lib.config import *



class FeatureExtractor(nn.Module):

    def __init__(self,):
        super(FeatureExtractor, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

        )

        self.linear = nn.Sequential(
            nn.Linear(4096, 300, bias=True).to(device),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def featureExtractor(**kwargs):
    return FeatureExtractor(**kwargs)
