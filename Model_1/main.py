import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from models.mtae import MTAE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math


from lib.config import *
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)

classes = {0: 'T-shirt',
                1: 'Trouser',
                2: 'Pullover',
                3: 'Dress',
                4: 'Coat',
                5: 'Sandal',
                6: 'Shirt',
                7: 'Sneaker',
                8: 'Bag',
                9: 'Ankle boot'}

cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
def load_rotated_fmnist(left_out_idx=0):
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    random = []
    for i in range(10):
        random.append(np.random.permutation(6000))

    for i in range(6):
        angle = 360 - 15 * i
        transform = transforms.Compose([transforms.RandomRotation(degrees=(angle, angle)), transforms.ToTensor()])
        fmnist_train = datasets.FashionMNIST(root='data/', download=False, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=fmnist_train, batch_size=60000, shuffle=False)

        a = next(iter(train_loader))

        targets = a[1]

        data = a[0]

        data_x = []
        data_y = []
        for j in range(10):
            idx = targets == j
            jth_target = targets[idx].to(device)
            jth_data = data[idx].to(device)
            jth_data = jth_data[random[j]]

            sample_x = jth_data[:200]
            sample_y = jth_target[:200]

            if i != left_out_idx and j != 0 and j != 5:
                data_x.append(sample_x)
                data_y.append(sample_y.float())



            if i==left_out_idx:
                data_x.append(sample_x)
                data_y.append(sample_y.float())



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

    train_x,train_y,test_x,test_y = load_rotated_fmnist(left_out_idx=dom)

    batch_size=50
    print('------------------------')
    mtae = MTAE()
    for epoch in range(100):
        mtae.epoch = epoch
        x_train = []
        y_train = []

        random = np.random.permutation(1600)

        for i in range(5):
            x = train_x[i]
            x_permuted = x[random]

            y = train_y[i]
            y_permuted = y[random]

            x_train.append(x_permuted)
            y_train.append(y_permuted)

        x_train = torch.cat(x_train).to(device)
        x_train = x_train.view(5,1600,28*28)

        y_train  = torch.cat(y_train).to(device)
        y_train = y_train.view(5,1600).long()


        for i in range(5):
            for j in range(5):
                avg_cost = 0
                for k in range(0,1600,50):

                    left_x = x_train[i][k:k+50,:]
                    labels = y_train[i][k:k+50]
                    right_x = x_train[j][k:k+50,:]
                    avg_cost+= mtae.train(left_x,right_x,labels,i)

                avg_cost = avg_cost/32

        print(avg_cost)
        print("----------------------------------------")

        x_train = x_train.view(8000,28*28)

        if (epoch+1)%10==0:
            criterion = nn.MSELoss()
            model = mtae.Encoder

            model.eval()
            with torch.no_grad():
                x_train = x_train.view((8000, 28 * 28))
                y_train = y_train.view(8000).long()
                out = model(x_train)

                pca_model = PCA()
                X_embedded = pca_model.fit_transform(out.cpu())
                print('Variance Ratio:', pca_model.explained_variance_ratio_[:2])

                ix1 = X_embedded[:, 0]
                ix2 = X_embedded[:, 1]

                fig, ax = plt.subplots()
                for g in np.unique(y_train.cpu()):
                    ix = np.where(y_train.cpu() == g)
                    ax.scatter(ix1[ix],ix2[ix],c=cmap[g], label=clothes[g],s=8)
                ax.legend()
                plt.show()


                predicted = []
                correct = 0
                for i in range(len(out)):
                    cost = []
                    for j in range(10):
                        cost.append(criterion(out[i], w2v[j]).item())


                    pred = cost.index(min(cost))
                    if pred == y_train[i]:
                        correct += 1
                        predicted.append(pred)

                accuracy = correct / len(y_train)
                print('Train Accuracy of the model: {} %'.format(accuracy))

                x_test = test_x.view((2000, 28 * 28))
                y_test = test_y.view(2000).long()
                out = model(x_test)

                predicted = []
                correct = 0
                correct0 = 0
                correct5 = 0
                for i in range(len(out)):
                    cost = []
                    for j in range(10):
                        cost.append(criterion(out[i], w2v[j]).item())


                    pred = cost.index(min(cost))

                    if pred == y_test[i] and y_test[i]==0:
                        correct0 += 1
                    elif pred == y_test[i] and y_test[i]==5:
                        correct5 += 1
                    elif pred==y_test[i]:
                        correct += 1

                    predicted.append(pred)

                accuracy = correct / (len(y_test)-400)
                print('Test Accuracy of the model: {} %'.format(accuracy))

                print('Correct', correct)
                print('Correct0:', correct0)
                print('Correct5:', correct5)
                accuracy = correct0 /200
                print('Test Accuracy of the model on 0: {} %'.format(accuracy))

                accuracy = correct5 / 200
                print('Test Accuracy of the model on 5: {} %'.format(accuracy))


for dom in range(1, 2):
    print("----------------------Domain.{}---------------------".format(dom))
    domain_specific_training(dom)
