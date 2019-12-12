import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
from models.mtae import MTAE
from models.mtae import Encoder
import math
from lib.config import *
from lib.utils import PCA_TSNE
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)


classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_rotated_cifar(left_out_idx):
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    random = []
    for i in range(10):
        random.append(np.random.permutation(5000))

    for i in range(6):
        angle = 360 - 15 * i
        transform = transforms.Compose([transforms.RandomRotation(degrees=(angle, angle)), transforms.ToTensor()])
        cifar_train = datasets.CIFAR10(root='data/', download=False, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=cifar_train, batch_size=50000, shuffle=False)

        full_data = next(iter(train_loader))

        targets = full_data[1]

        data = full_data[0]

        data_x = []
        data_y = []
        for j in range(10):
            idx = targets == j
            jth_target = targets[idx].to(device)
            jth_data = data[idx].to(device)
            jth_data = jth_data[random[j]]


            sample_x = jth_data[:400]
            sample_y = jth_target[:400]

            if i != left_out_idx:
                data_x.append(sample_x)
                data_y.append(sample_y)

            if i==left_out_idx:
                data_x.append(jth_data)
                data_y.append(jth_target)

        data_x = torch.cat(data_x).to(device)
        data_y = torch.cat(data_y).to(device)


        if i != left_out_idx:
            train_x.append(data_x)
            train_y.append(data_y)
        else:
            test_x = data_x
            test_y = data_y

    return train_x, train_y, test_x, test_y


def domain_specific_training(dom):
    train_x, train_y, test_x, test_y = load_rotated_cifar(dom)
    length_of_domain = len(train_x[0])

    print(length_of_domain)
    print(train_x[0].shape, train_y[0].shape, test_x.shape, test_y.shape)



    batch_size=50
    print('------------------------')
    mtae = MTAE()
    for epoch in range(100):
        x_train = []
        y_train = []

        random = np.random.permutation(length_of_domain)

        for i in range(5):
            x = train_x[i]
            x_permuted = x[random]

            y = train_y[i]
            y_permuted = y[random]

            x_train.append(x_permuted)
            y_train.append(y_permuted)

        x_train = torch.cat(x_train).to(device)
        x_train = x_train.view(5,length_of_domain,3,32,32)

        y_train  = torch.cat(y_train).to(device)
        y_train = y_train.view(5,length_of_domain)


        for i in range(5):
            for j in range(5):
                avg_cost = 0
                for k in range(0,length_of_domain,batch_size):

                    left_x = x_train[i][k:k+batch_size,:,:,:]
                    labels = y_train[i][k:k+batch_size]
                    right_x = x_train[j][k:k+batch_size,:,:,:]
                    avg_cost+= mtae.train(left_x,right_x,labels,i)

                avg_cost = avg_cost/(length_of_domain/batch_size)

        print(avg_cost)



        if (epoch+1)%25==0:
            print('After {} epochs'.format(epoch))

            criterion = nn.MSELoss()
            encoder = mtae.Encoder
            encoder.eval()


            with torch.no_grad():
                x_train = x_train.view(5*length_of_domain, 3,32,32)
                y_train = y_train.view(5*length_of_domain).long()
                train_out = encoder(x_train)

                # PCA_TSNE(np.array(train_out.cpu()),np.array(y_train.cpu()))

                predicted = []
                correct = 0
                for i in range(len(y_train)):
                    cost = []
                    for j in range(10):
                        cost.append(criterion(train_out[i], w2v[j]).item())

                    pred = cost.index(min(cost))
                    predicted.append(pred)
                    if pred == y_train[i]:
                        correct += 1

                accuracy = correct / len(y_train)
                print('Train Accuracy of the model: {} %'.format(accuracy))
                del train_out

            with torch.no_grad():

                x_test = test_x.view((50000,3,32,32)).to(device)
                y_test = test_y.view(50000).to(device)

                test_tensor = torch.utils.data.TensorDataset(x_test, y_test)
                test_loader = torch.utils.data.DataLoader(dataset=test_tensor, batch_size=1000, shuffle=False)

                y_test = []
                test_out = []
                for i, (X, Y) in enumerate(test_loader):
                    out = encoder(X)
                    test_out.append(out)
                    y_test.append(Y)


                test_out = torch.cat(test_out).to(device)
                y_test = torch.cat(y_test).to(device)

                # PCA_TSNE(np.array(test_out.cpu()),np.array(y_test.cpu()))

                predicted = []
                correct = 0
                for i in range(len(y_test)):
                    cost = []
                    for j in range(10):
                        cost.append(criterion(test_out[i], w2v[j]).item())

                    pred = cost.index(min(cost))
                    predicted.append(pred)
                    if pred == y_test[i]:
                        correct += 1

                accuracy = correct / len(y_test)
                print('Test Accuracy of the model: {} %'.format(accuracy))
                del test_out

for dom in range(6):
    print("----------------------Domain.{}---------------------".format(dom))
    domain_specific_training(dom)
