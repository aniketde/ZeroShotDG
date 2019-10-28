from .encoder import Encoder
from .decoder import Decoder
import torch
import torch.nn as nn
from lib.config import *


class MTAE:
    def __init__(self):
        super(MTAE, self).__init__()

        learning_rate = 0.0003
        weight_decay = 5e-6

        self.Encoder = Encoder().to(device)

        # self.Encoder = self.Encoder.train()

        # self.Decoder = Decoder().to(device)
        # self.params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())

        # self.optimizer = torch.optim.Adamax(params=self.params, lr=learning_rate,weight_decay=weight_decay)

        self.decoders = []
        self.optimizers = []
        for i in range(5):
            self.decoders.append(Decoder().to(device))
            self.params = list(self.Encoder.parameters()) + list(self.decoders[i].parameters())
            self.optimizers.append(torch.optim.RMSprop(params=self.params, lr=learning_rate, weight_decay=weight_decay))

        self.criterion = nn.MSELoss()
        self.criterion2 = nn.MSELoss()


        self.vector = w2v
        self.epoch = 0



    def train(self, X, Y, labels, domainId):
        self.optimizers[domainId].zero_grad()
        # self.optimizer.zero_grad()

        out1 = self.Encoder(X)

        loss1 = self.criterion(out1, self.vector[labels])
        # out2 = self.Decoder(out1)

        out2 = self.decoders[domainId](out1)
        loss2 = self.criterion(out2, Y)

        cost = loss1 + loss2

        cost.backward()

        # self.optimizer.step()
        # self.optimizer.zero_grad()

        self.optimizers[domainId].step()
        self.optimizers[domainId].zero_grad()

        return cost.data.item()
