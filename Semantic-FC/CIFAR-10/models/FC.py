from .classifier import Classifier,Critic_Network_MLP
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


class FeatureCritic:
    def __init__(self):
        super(FeatureCritic, self).__init__()

        learning_rate = 0.0001
        weight_decay = 0.00005

        #RMSPROP
        # learning_rate = 0.0003
        # weight_decay = 5e-6

        self.feature_extractor = featureExtractor().to(device)
        self.feature_extractor.apply(Xavier)

        self.epoch = 0

        self.Classifier = Classifier().to(device)
        self.Classifier.apply(Xavier)

        self.Critic_Network_MLP = Critic_Network_MLP().to(device)
        self.Critic_Network_MLP.apply(Xavier)

        self.params = list(self.feature_extractor.parameters())
        self.optimizer_interim_feature_extractor = torch.optim.Adamax(params=self.params, lr=learning_rate, weight_decay=0.00001)
        self.optimizer_final_feature_extractor = torch.optim.Adamax(params=self.params, lr=learning_rate,weight_decay=0.00001)

        self.params = list(self.Classifier.parameters())
        self.optimizer_interim_classifier = torch.optim.Adamax(params=self.params, lr=learning_rate, weight_decay=0)
        self.optimizer_final_classifier = torch.optim.Adamax(params=self.params, lr=learning_rate, weight_decay=0)

        self.params = list(self.Critic_Network_MLP.parameters())
        self.optimizer_regularizer = torch.optim.Adamax(params=self.params, lr=learning_rate)

        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.vector = w2v


    def train(self, D_x_trn,D_y_trn,D_x_val, D_y_val):
        length_of_domain = len(D_y_val[0])

        batch_size = 50
        for k in range(0,length_of_domain,batch_size):

            self.optimizer_interim_feature_extractor.zero_grad()
            self.optimizer_interim_classifier.zero_grad()
            self.optimizer_regularizer.zero_grad()

            unmodified_feature_extractor = self.feature_extractor.state_dict()
            unmodified_classifier = self.Classifier.state_dict()

            loss_mse = 0
            loss_ce = 0
            loss_aux = 0
            for i in range(len(D_x_trn)):
                features = self.feature_extractor(D_x_trn[i][k:k+50])
                out = self.Classifier(features)

                loss_mse += self.criterion_mse(features,self.vector[D_y_trn[i][k:k+50]])
                loss_ce += self.criterion_ce(out, D_y_trn[i][k:k+50])
                loss_aux += self.Critic_Network_MLP(features)

            # loss = loss_ce
            loss = loss_mse + loss_ce
            loss.backward(retain_graph=True)
            self.optimizer_interim_feature_extractor.step()
            self.optimizer_interim_classifier.step()

            self.optimizer_interim_feature_extractor.zero_grad()
            self.optimizer_interim_classifier.zero_grad()

            old_feature_extractor = self.feature_extractor.state_dict()
            old_classifier = self.Classifier.state_dict()

            loss_aux.backward(retain_graph=True)

            self.optimizer_interim_feature_extractor.step()
            self.optimizer_interim_classifier.step()

            self.optimizer_interim_feature_extractor.zero_grad()
            self.optimizer_interim_classifier.zero_grad()

            self.optimizer_regularizer.zero_grad()

            loss_new_val = []
            meta_loss = 0
            for i in range(len(D_x_val)):
                features = self.feature_extractor(D_x_val[i][k:k + 50])
                out = self.Classifier(features)
                loss_new_val.append(self.criterion_ce(out, D_y_val[i][k:k + 50]) + self.criterion_mse(features,self.vector[D_y_val[i][k:k+50]]))

            self.feature_extractor.load_state_dict(old_feature_extractor)
            self.Classifier.load_state_dict(old_classifier)

            loss_old_val = []
            for i in range(len(D_x_val)):
                features = self.feature_extractor(D_x_val[i][k:k + 50])
                out = self.Classifier(features)
                loss_old_val.append(self.criterion_ce(out, D_y_val[i][k:k + 50]) + self.criterion_mse(features,self.vector[D_y_val[i][k:k+50]]))

            for i in range(len(D_x_val)):

                reward =  loss_new_val[i]-loss_old_val[i]
                # calculate the updating rule of omega, here is the max function of h.
                utility = torch.tanh(reward)
                # so, here is the min value transfering to the backpropogation.
                loss_held_out = utility.sum()
                meta_loss += loss_held_out

            meta_loss.backward()
            self.optimizer_regularizer.step()

            self.feature_extractor.load_state_dict(unmodified_feature_extractor)
            self.Classifier.load_state_dict(unmodified_classifier)

            self.optimizer_final_feature_extractor.zero_grad()
            self.optimizer_final_classifier.zero_grad()

            loss_ce.backward(retain_graph=True)
            self.optimizer_final_classifier.step()

            loss_aux.backward()
            self.optimizer_final_feature_extractor.step()

        print(loss_aux)