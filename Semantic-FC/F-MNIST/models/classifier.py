import torch.nn as nn
from lib.config import *
from lib.utils import Unflatten
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(300, 10),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.classifier(x)
        return out

class Critic_Network_MLP(nn.Module):
    def __init__(self, h=300, hh=512):
        super(Critic_Network_MLP, self).__init__()
        self.fc1 = nn.Linear(h, hh)
        self.fc2 = nn.Linear(hh, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = nn.functional.softplus(self.fc2(x))
        return torch.mean(x)