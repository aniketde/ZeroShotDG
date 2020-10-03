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
datasets.FashionMNIST(root='../../data/', download=False, train=True)

classes = ['Tshirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

def load_rotated_fmnist(left_out_idx):
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
        fmnist_train = datasets.FashionMNIST(root='../../data/', download=False, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=fmnist_train, batch_size=60000, shuffle=False)

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

def domain_specific_training(dom,clas):
    train_x, train_y, test_x, test_y = load_rotated_fmnist(dom)
    length_of_domain = len(train_x[0])

    train_rst = []
    train_ms_rst = []
    zs_rst = []
    zs_mse_rst = []
    test_rst = []
    test_mse_rst = []

    for i in range(len(train_x)):
        for k in clas:
            idx = train_y[i] != k
            train_y[i] = train_y[i][idx]
            train_x[i] = train_x[i][idx]

    zs_y = []
    zs_x = []
    for k in clas:
        idx = test_y ==k
        zs_x.append(test_x[idx])
        zs_y.append(test_y[idx])

        idx = test_y != k
        test_y = test_y[idx]
        test_x = test_x[idx]

    zs_x = torch.cat(zs_x)
    zs_y = torch.cat(zs_y)

    train_x = torch.cat(train_x)
    train_y = torch.cat(train_y)

    train_x = train_x.view(5, length_of_domain-800, 1, 28, 28)
    train_y = train_y.view(5, length_of_domain-800).long()

    length_of_domain -= 800

    batch_size=50
    print('------------------------')
    mtae = MTAE()
    for epoch in range(120):
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
        x_train = x_train.view(5,length_of_domain,1,28,28)

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

        if (epoch+1)%15==0:

            print('After {} epochs'.format(epoch))

            criterion = nn.MSELoss()
            encoder = mtae.Encoder
            encoder.eval()

            with torch.no_grad():
                x_train = x_train.view(5 * length_of_domain, 1, 28, 28)
                y_train = y_train.view(5 * length_of_domain).long()

                train_tensor = torch.utils.data.TensorDataset(x_train, y_train)
                train_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=3000, shuffle=False)

                y_train = []
                train_out = []
                for i, (X, Y) in enumerate(train_loader):
                    feat = encoder(X)
                    train_out.append(feat)
                    y_train.append(Y)

                train_out = torch.cat(train_out).to(device)
                y_train = torch.cat(y_train).to(device)

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
                print('Train MSE Accuracy of the model: {} %'.format(accuracy))
                train_ms_rst.append(accuracy)

            with torch.no_grad():
                zs_x = zs_x.view(12000, 1, 28, 28)
                zs_y = zs_y.view(12000).long()

                zs_tensor = torch.utils.data.TensorDataset(zs_x, zs_y)
                zs_loader = torch.utils.data.DataLoader(dataset=zs_tensor, batch_size=3000, shuffle=False)

                y_zs = []

                zs_out = []
                for i, (X, Y) in enumerate(zs_loader):
                    feat = encoder(X)
                    zs_out.append(feat)
                    y_zs.append(Y)

                zs_out = torch.cat(zs_out).to(device)
                y_zs = torch.cat(y_zs).to(device)

                # PCA_TSNE(np.array(train_out.cpu()),np.array(y_train.cpu()))

                predicted = []
                correct = 0
                for i in range(len(y_zs)):
                    cost = []
                    for j in range(10):
                        cost.append(criterion(zs_out[i], w2v[j]).item())
                    pred = cost.index(min(cost))
                    predicted.append(pred)
                    if pred == y_zs[i]:
                        correct += 1
                # print(predicted)
                accuracy = correct / len(y_zs)
                print('ZS MSE Accuracy of the model: {} %'.format(accuracy))

                predicted = []
                correct = 0
                correct1 = 0
                correct2 = 0
                for i in range(len(y_zs)):
                    cost = []
                    for j in clas:
                        cost.append(criterion(zs_out[i], w2v[j]).item())
                    # if i%500==0:
                    #     print(cost)
                    pred = clas[cost.index(min(cost))]
                    predicted.append(pred)
                    if pred == y_zs[i]:
                        correct += 1
                        if y_zs[i] == clas[0]:
                            correct1 += 1
                        else:
                            correct2 += 1

                # print(predicted)
                accuracy = correct / len(y_zs)
                print('ZS MSE Accuracy of the model: {} %'.format(accuracy))
                print('ZS MSE Accuracy of the model: {}.;;;{}:{} %'.format(accuracy,clas[0],correct1/5000))
                print('ZS MSE Accuracy of the model: {},;;;{}:{} %'.format(accuracy,clas[1],correct2/5000))
                zs_mse_rst.append(accuracy)

                print('ZS OUT TSNE')
                PCA_TSNE(np.array(zs_out.cpu()), np.array(y_zs.cpu()), 'zs_' + str(clas) + str(epoch))

                print('ZS OUT and TRAIN OUT TSNE')
                PCA_TSNE(np.array(torch.cat((train_out, zs_out), 0).cpu()),(np.array(torch.cat((y_train, y_zs), 0).cpu())), 'zs_trn' + str(clas) + str(epoch))

                # PCA_TSNE(np.array(torch.cat((train_out, zs_out), 0).cpu()),(np.array(torch.cat((y_train, y_zs), 0).cpu())))

            with torch.no_grad():

                x_test = test_x.view((48000, 1, 28, 28)).to(device)
                y_test = test_y.view(48000).to(device)

                test_tensor = torch.utils.data.TensorDataset(x_test, y_test)
                test_loader = torch.utils.data.DataLoader(dataset=test_tensor, batch_size=3000, shuffle=False)

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
                print('Test MSE Accuracy of the model: {} %'.format(accuracy))
                test_mse_rst.append(accuracy)
                del test_out

    return train_rst, train_ms_rst, zs_rst, zs_mse_rst, test_rst, test_mse_rst


classes = ['Tshirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

seeds = [107,109,997,991,804,451,321,652,854,102]
for dom in range(3,4):
    import statistics

    zs = [[0, 5], [5, 6], [0, 9], [5,9], [0,7]]
    for clas in zs:
        print(clas)

    for clas in zs:
        final_zs = []
        final_test_mse = []
        final_test = []

        print("----------------------Domain.{}.{}---------------------".format(dom, clas))

        for repeat in range(1):
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(seeds[repeat])
            torch.manual_seed(seeds[repeat])
            np.random.seed(seeds[repeat])

            train_rst,train_ms , zs_rst,zs_mse, test_rst,test_mse = domain_specific_training(dom,clas=clas)
            print('Train MSE :', max(train_ms))
            print('ZS MSE : ',max(zs_mse))
            final_zs.append(max(zs_mse))

            print('Test MSE : ', max(test_mse))
            final_test_mse.append(max(test_mse))

        print('Zero Shot' , final_zs,'\n Mean:', statistics.mean(final_zs),'\n Std Dev',statistics.stdev(final_zs) )
        print('Test MSE :', final_test_mse,'\n Mean:', statistics.mean(final_test_mse), '\n Std Dev',statistics.stdev(final_test_mse))

