import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(300, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)


    def forward(self, x):
        x = F.relu(self.fc1(x)).to(device)
        x = F.log_softmax(self.fc2(x)).to(device)
        return x
