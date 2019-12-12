import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from models.mtae import MTAE
from models.neuralNet import NeuralNet
from utils import *

torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)


cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
def load_rotated_cifar(left_out_idx):
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    random = []
    for i in range(10):
        random.append(np.random.permutation(5000))

    for i in range(6):
        angle = 360- 15 * i
        transform = transforms.Compose([transforms.RandomRotation(degrees=(angle, angle)), transforms.ToTensor()])
        cifar_train = datasets.CIFAR10(root='data/', download=False, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=cifar_train, batch_size=60000, shuffle=False)

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


def Neuron(feat_train, y_train, feat_test, y_test):

    model = NeuralNet().to(device)

    learning_rate = 0.001
    weight_decay = 0.00001

    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    batch_size = 40

    training_epochs = 150

    train_tensor = torch.utils.data.TensorDataset(feat_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True)

    total_batch = len(y_train) // batch_size

    for epoch in range(training_epochs):
        avg_cost = 0
        for i, (X, Y) in enumerate(train_loader):
            model.zero_grad()

            hypothesis = model(X)

            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()
            model.zero_grad()
            avg_cost += (cost.data / total_batch)

        if (epoch+1) % 25 == 0:
            print("Epoch: {}, averaged cost = {:.4f}".format(epoch + 1, avg_cost.item()))

            with torch.no_grad():
                out = model(feat_train)

                _, predicted = torch.max(out, dim=1)
                correct = sum(np.array((predicted == y_train).cpu()))
                accuracy = correct / len(y_train)
                print('Train Accuracy of the model: {} %'.format(accuracy))

            with torch.no_grad():
                out = model(feat_test)
                _, predicted = torch.max(out, dim=1)
                correct = sum(np.array((predicted == y_test).cpu()))
                accuracy = correct / len(y_test)
                print('Test Accuracy of the model: {} %'.format(accuracy))



def linear_svm(feat_train, y_train, feat_test, y_test):  # training_data,test_data):


    lin_svm = OneVsRestClassifier(LinearSVC(C=1, loss='hinge', max_iter=50000), n_jobs=-1)

    lin_svm.fit(feat_train, y_train)

    train_outcome = lin_svm.predict(feat_train)

    correct = sum(np.array(train_outcome == y_train))

    train_accuracy = correct / len(y_train)
    print(train_outcome)
    print(y_train)
    print("Training Accuracy: %.8f" % train_accuracy)

    test_outcome = lin_svm.predict(np.array(feat_test))
    print('Outcome', test_outcome)
    print('Actual', y_test)
    correct = sum(np.array(test_outcome == y_test))
    accuracy = correct/ len(test_outcome)

    print("Test accuracy: %.8f" % accuracy)

    return accuracy


def domain_specific_training(dom):
    train_x,train_y,test_x,test_y = load_rotated_cifar(dom)

    length_of_domain = len(train_x[0])
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

    model = mtae.Encoder
    model.eval()

    with torch.no_grad():
        x_train = x_train.view((5*length_of_domain, 3,32 ,32))
        y_train = y_train.view(5*length_of_domain)

        train_tensor = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=1000, shuffle=False)

        feat_train = []
        y_train = []
        for i, (X, Y) in enumerate(train_loader):
            out = model(X)
            feat_train.append(out)
            y_train.append(Y)

        feat_train = torch.cat(feat_train).to(device)
        y_train = torch.cat(y_train).to(device)



    with torch.no_grad():
        x_test = test_x.view((len(test_y), 3,32, 32))
        y_test = test_y.view(len(test_y))

        test_tensor = torch.utils.data.TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_tensor, batch_size=1000, shuffle=False)

        feat_test = []
        y_test = []
        for i, (X, Y) in enumerate(test_loader):
            out = model(X)
            feat_test.append(out)
            y_test.append(Y)

        feat_test = torch.cat(feat_test).to(device)
        y_test = torch.cat(y_test).to(device)

    linear_svm(np.array(feat_train.cpu()),np.array(y_train.cpu()),np.array(feat_test.cpu()),np.array(y_test.cpu()))
    Neuron(feat_train.detach(), y_train.detach(), feat_test.detach(), y_test.detach())



for dom in range(6):
    print("----------------------Domain.{}, Class---------------------".format(dom))
    domain_specific_training(dom)
