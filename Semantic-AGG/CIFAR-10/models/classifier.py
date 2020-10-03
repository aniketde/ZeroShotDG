import torch.nn as nn
from lib.config import *
from lib.utils import Unflatten
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(300, 10)
        ).to(device)

    def forward(self, x):
        out = self.classifier(x).to(device)
        return out