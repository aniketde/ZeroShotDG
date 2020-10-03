from .classifier import Classifier
import torch
import torch.nn as nn
from lib.config import *
import math
from .featureExtractor import *

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class AGG:
    def __init__(self):
        super(AGG, self).__init__()

        learning_rate = 0.0001
        weight_decay = 0.00005

        #RMSPROP
        # learning_rate = 0.0003
        # weight_decay = 5e-6

        self.feature_extractor = featureExtractor().to(device)
        self.epoch = 0
        self.Classifier = Classifier().to(device)
        self.Classifier.apply(Xavier)
        self.params = list(self.feature_extractor.parameters()) + list(self.Classifier.parameters())
        self.optimizer = torch.optim.Adamax(params=self.params, lr=learning_rate, weight_decay=weight_decay)

        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.vector = w2v


    def train(self, X, Y, domainId):
        self.optimizer.zero_grad()

        out1 = self.feature_extractor(X)
        loss1 = self.criterion_mse(out1, self.vector[Y])

        out2 = self.Classifier(out1)
        loss2 = self.criterion_ce(out2, Y)

        cost = loss1 + loss2

        cost.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return cost.data.item()
