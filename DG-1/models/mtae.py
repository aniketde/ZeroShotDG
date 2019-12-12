from .encoder import Encoder
from .decoder import Decoder
import torch
import torch.nn as nn
import math

from utils import *

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MTAE:
    def __init__(self):
        super(MTAE, self).__init__()

        learning_rate = 0.001
        weight_decay = 5e-6

        #RMSPROP
        # learning_rate = 0.0003
        # weight_decay = 5e-6

        self.Encoder = Encoder().to(device)
        self.Encoder.apply(Xavier)

        self.decoders = []
        self.optimizers = []
        for i in range(5):
            self.decoders.append(Decoder().to(device))
            self.decoders[i].apply(Xavier)
            self.params = list(self.Encoder.parameters()) + list(self.decoders[i].parameters())
            self.optimizers.append(torch.optim.Adam(params=self.params, lr=learning_rate, weight_decay=weight_decay))

        self.criterion = nn.MSELoss()



    def train(self, X, Y, labels, domainId):
        self.optimizers[domainId].zero_grad()

        out = self.Encoder(X)

        out = self.decoders[domainId](out)
        cost = self.criterion(out, Y)

        cost.backward()

        self.optimizers[domainId].step()
        self.optimizers[domainId].zero_grad()

        return cost.data.item()
