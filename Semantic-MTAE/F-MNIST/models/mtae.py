from .encoder import Encoder
from .decoder import Decoder
import torch
import torch.nn as nn
from lib.config import *
import math
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

        learning_rate = 0.0001
        weight_decay = 5e-6

        self.Encoder = Encoder().to(device)
        self.Encoder.apply(Xavier)

        self.decoders = []
        self.optimizers = []
        for i in range(5):
            self.decoders.append(Decoder().to(device))
            self.decoders[i].apply(Xavier)
            self.params = list(self.Encoder.parameters()) + list(self.decoders[i].parameters())
            self.optimizers.append(torch.optim.Adamax(params=self.params, lr=learning_rate, weight_decay=weight_decay))


        self.criterion = nn.MSELoss()

        self.vector = w2v

    def train(self, X, Y, labels, domainId):
        self.optimizers[domainId].zero_grad()


        out1 = self.Encoder(X)
        loss1 = self.criterion(out1, self.vector[labels])


        out2 = self.decoders[domainId](out1)
        loss2 = self.criterion(out2, Y)

        cost = loss1 + loss2

        cost.backward()

        self.optimizers[domainId].step()
        self.optimizers[domainId].zero_grad()

        return cost.data.item()