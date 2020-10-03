import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
from models.FC import FeatureCritic
import math
from lib.config import *
from lib.utils import PCA_TSNE
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
import random
random.seed(seed)
datasets.FashionMNIST(root='../../data/', download=False, train=True)

import matplotlib.pyplot as plt

classes = ['Tshirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

def load_rotated_fmnist(left_out_idx):
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    random_perm = []
    for i in range(10):
        random_perm.append(np.random.permutation(6000))

    for i in range(6):
        angle = 360 - 15 * i
        transform = transforms.Compose([transforms.RandomRotation(degrees=(angle, angle)), transforms.ToTensor()])
        fmnist_train = datasets.FashionMNIST(root='../../data/', download=False, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=fmnist_train, batch_size=60000, shuffle=False,num_workers = 0,worker_init_fn=random.seed(1))

        full_data = next(iter(train_loader))

        targets = full_data[1]

        data = full_data[0]

        data_x = []
        data_y = []
        for j in range(10):
            idx = targets == j
            jth_target = targets[idx].to(device)
            jth_data = data[idx].to(device)
            jth_data = jth_data[random_perm[j]]


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

    length_of_domain-=800

    batch_size=50
    print('------------------------')

    model = FeatureCritic()
    for epoch in range(200):
        x_train = []
        y_train = []

        random_p = np.random.permutation(length_of_domain)

        for i in range(5):
            x = train_x[i]
            x_permuted = x[random_p]

            y = train_y[i]
            y_permuted = y[random_p]

            x_train.append(x_permuted)
            y_train.append(y_permuted)

        trn = random.sample(range(5), 3)

        D_x_trn = []
        D_y_trn = []
        D_x_val = []
        D_y_val = []

        for i in trn:
            D_x_trn.append(x_train[i])
            D_y_trn.append(y_train[i])

        for i in range(5):
            if i not in trn:
                D_x_val.append(x_train[i])
                D_y_val.append(y_train[i])

        model.train(D_x_trn,D_y_trn,D_x_val,D_y_val)

        x_train = torch.cat(x_train).to(device)
        x_train = x_train.view(5,length_of_domain,1,28,28)

        y_train  = torch.cat(y_train).to(device)
        y_train = y_train.view(5,length_of_domain)

        if (epoch+1)%25==0:
            print('After {} epochs'.format(epoch))

            criterion = nn.MSELoss()

            feature_extractor = model.feature_extractor
            feature_extractor.eval()

            classifier = model.Classifier
            classifier.eval()

            with torch.no_grad():

                x_train = x_train.view(5 * length_of_domain, 1, 28, 28)
                y_train = y_train.view(5 * length_of_domain).long()

                train_tensor = torch.utils.data.TensorDataset(x_train, y_train)
                train_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=500, shuffle=False,num_workers = 0,worker_init_fn=random.seed(1))

                y_train = []
                train_out = []
                classifier_out = []
                for i, (X, Y) in enumerate(train_loader):
                    feat = feature_extractor(X)
                    out = classifier(feat)
                    train_out.append(feat)
                    classifier_out.append(out)
                    y_train.append(Y)

                train_out = torch.cat(train_out).to(device)
                y_train = torch.cat(y_train).to(device)
                classifier_out = torch.cat(classifier_out).to(device)

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

                _, predicted = torch.max(classifier_out, dim=1)
                correct = sum(np.array((predicted == y_train).cpu()))

                accuracy = correct / len(y_train)
                print('Train Accuracy of the model: {} %'.format(accuracy))

                train_rst.append(accuracy)

            with torch.no_grad():
                zs_x = zs_x.view(12000, 1, 28, 28)
                zs_y = zs_y.view(12000).long()

                zs_tensor = torch.utils.data.TensorDataset(zs_x, zs_y)
                zs_loader = torch.utils.data.DataLoader(dataset=zs_tensor, batch_size=500, shuffle=False,num_workers = 0,worker_init_fn=random.seed(1))

                y_zs = []
                zs_out = []
                zs_classifier_out = []
                for i, (X, Y) in enumerate(zs_loader):
                    feat = feature_extractor(X)
                    out = classifier(feat)
                    zs_out.append(feat)
                    zs_classifier_out.append(out)
                    y_zs.append(Y)

                zs_out = torch.cat(zs_out).to(device)
                y_zs = torch.cat(y_zs).to(device)
                zs_classifier_out = torch.cat(zs_classifier_out).to(device)

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

                accuracy = correct / len(y_zs)
                print('ZS MSE Accuracy of the model: {} %'.format(accuracy))

                predicted = []
                correct = 0
                correct1 = 0
                correct2=0
                for i in range(len(y_zs)):
                    cost = []
                    for j in clas:
                        cost.append(criterion(zs_out[i], w2v[j]).item())
                    pred = clas[cost.index(min(cost))]
                    predicted.append(pred)
                    if pred == y_zs[i]:
                        correct += 1
                        if y_zs[i]==clas[0]:
                            correct1+=1
                        else:
                            correct2+=1

                accuracy = correct / len(y_zs)
                print('ZS MSE Accuracy of the model: {} %'.format(accuracy))
                print('ZS MSE Accuracy of the model: {}.;;;{}:{} %'.format(accuracy,clas[0],correct1/5000))
                print('ZS MSE Accuracy of the model: {},;;;{}:{} %'.format(accuracy,clas[1],correct2/5000))

                zs_mse_rst.append(accuracy)

                _, predicted = torch.max(zs_classifier_out, dim=1)
                correct = sum(np.array((predicted == y_zs).cpu()))

                accuracy = correct / len(y_zs)
                print('ZS Accuracy of the model: {} %'.format(accuracy))
                zs_rst.append(accuracy)


            with torch.no_grad():

                x_test = test_x.view((48000,1,28,28)).to(device)
                y_test = test_y.view(48000).to(device)

                test_tensor = torch.utils.data.TensorDataset(x_test, y_test)
                test_loader = torch.utils.data.DataLoader(dataset=test_tensor, batch_size=1000, shuffle=False,num_workers = 0,worker_init_fn=random.seed(1))

                y_test = []
                test_out = []
                classifier_out = []
                for i, (X, Y) in enumerate(test_loader):
                    out = feature_extractor(X)
                    test_out.append(out)
                    classifier_out.append(classifier(out))
                    y_test.append(Y)

                test_out = torch.cat(test_out).to(device)
                y_test = torch.cat(y_test).to(device)
                classifier_out = torch.cat(classifier_out).to(device)

                _, predicted = torch.max(classifier_out, dim=1)
                correct = sum(np.array((predicted == y_test).cpu()))
                accuracy = correct / len(y_test)
                print('Test Accuracy of the model: {} %'.format(accuracy))

                test_rst.append(accuracy)


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
for dom in range(4,5):
    import statistics
    zs = [[5,9]]
    for clas in zs:
        print(clas)

    for clas in zs:

        final_zs = []
        final_test_mse = []
        final_test = []

        print("----------------------Domain.{}.{}---------------------".format(dom, clas))

        for repeat in range(5):
            torch.backends.cudnn.enabled=False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seeds[repeat])
            torch.manual_seed(seeds[repeat])
            np.random.seed(seeds[repeat])
            random.seed(seeds[repeat])

            train_rst,train_ms , zs_rst,zs_mse, test_rst,test_mse = domain_specific_training(dom,clas=clas)
            print('Train : ', max(train_rst))
            print('Train MSE :', max(train_ms))
            print('ZS : ', max(zs_rst))
            print('ZS MSE : ',max(zs_mse))
            final_zs.append(max(zs_mse))

            print('TEST : ', max(test_rst))
            final_test.append(max(test_rst))
            print('Test MSE : ', max(test_mse))
            final_test_mse.append(max(test_mse))

        print('Zero Shot' , final_zs,'\n Mean:', statistics.mean(final_zs),'\n Std Dev',statistics.stdev(final_zs) )
        print('Test :', final_test,'\n Mean:', statistics.mean(final_test), '\n Std Dev',statistics.stdev(final_test))
        print('Test MSE :', final_test_mse,'\n Mean:', statistics.mean(final_test_mse), '\n Std Dev',statistics.stdev(final_test_mse))
