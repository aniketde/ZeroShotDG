import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
from models.AGG import AGG
import matplotlib.pyplot as plt
import math
from lib.config import *
from lib.utils import PCA_TSNE
from PIL import Image
import torchvision.transforms as transforms
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
import os

classes = ['dog','elephant', 'giraffe','guitar', 'horse', 'house', 'person']
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_pacs(left_out_idx):

    domain_name = {'0': 'photo', '1': 'art_painting', '2': 'cartoon', '3': 'sketch'}

    data_folder = './data/PACS/pacs_label/'
    train_data = ['photo_train_kfold.txt',
                  'art_painting_train_kfold.txt',
                  'cartoon_train_kfold.txt',
                  'sketch_train_kfold.txt']

    val_data = ['photo_crossval_kfold.txt',
                'art_painting_crossval_kfold.txt',
                'cartoon_crossval_kfold.txt',
                'sketch_crossval_kfold.txt']

    test_data = ['photo_test_kfold.txt',
                 'art_painting_test_kfold.txt',
                 'cartoon_test_kfold.txt',
                 'sketch_test_kfold.txt']

    train_paths = []
    for data in train_data:
        path = os.path.join(data_folder, data)
        train_paths.append(path)

    val_paths = []
    for data in val_data:
        path = os.path.join(data_folder, data)
        val_paths.append(path)

    test_paths = []
    test_paths.append(train_paths[left_out_idx])
    test_paths.append(os.path.join(data_folder, test_data[left_out_idx]))

    train_paths.remove(train_paths[left_out_idx])
    val_paths.remove(val_paths[left_out_idx])

    train_images = []
    train_labels = []

    for train_path in train_paths:
        images = []
        labels = []
        zs_images = []
        zs_labels = []
        with open(train_path,'r') as file_to_read:

            line = file_to_read.readline()
            while line:
                image, label = [i for i in line.split()]
                image = transform(Image.open('data/PACS/pacs_data/'+image)).reshape(1,3,224,224)
                if int(label) == 5:
                    zs_images.append(image)
                    zs_labels.append(int(label)-1)
                else:
                    images.append(image)
                    labels.append(int(label)-1)
                line = file_to_read.readline()


        images = torch.cat(images).to(device)
        labels = torch.tensor(labels).to(device)
        train_images.append(images.detach())
        train_labels.append(labels.detach())
        del images, labels

    test_images = []
    test_labels = []

    for test_path in test_paths:
        images = []
        labels = []
        with open(test_path,'r') as file_to_read:

            line = file_to_read.readline()
            while line:
                image, label = [i for i in line.split()]
                image = transform(Image.open('data/PACS/pacs_data/'+image)).reshape(1,3,224,224)
                if int(label) == 5:
                    zs_images.append(image)
                    zs_labels.append(int(label)-1)
                else:
                    images.append(image)
                    labels.append(int(label)-1)
                line = file_to_read.readline()

        images = torch.cat(images)
        labels = torch.tensor(labels)
        test_images.append(images.detach())
        test_labels.append(labels)



        del images, labels

    zs_images = torch.cat(zs_images).to(device)
    zs_labels = torch.tensor(zs_labels).to(device)

    test_images = torch.cat(test_images).to(device)
    test_labels = torch.cat(test_labels).to(device)

    return train_images, train_labels, test_images, test_labels, zs_images, zs_labels

def domain_specific_training(dom):
    train_x, train_y, test_x, test_y, zs_x, zs_y = load_pacs(dom)
    length_of_domain = len(train_x[0])

    print(length_of_domain)
    print(train_x[0].shape, train_y[0].shape, test_x.shape, test_y.shape)


    batch_size=50


    domain_loader= []
    for i in range(3):
        domain_tensor = torch.utils.data.TensorDataset(train_x[i], train_y[i])
        domain_loader.append(torch.utils.data.DataLoader(dataset=domain_tensor, batch_size=batch_size, shuffle=True))

    zs_tensor = torch.utils.data.TensorDataset(zs_x, zs_y)
    zs_loader = torch.utils.data.DataLoader(dataset=zs_tensor, batch_size=batch_size, shuffle=False)


    print('------------------------')
    model = AGG()
    for epoch in range(200):
        model.epoch = epoch
        avg_cost = 0
        for i in range(3):
            for j, (X, Y) in enumerate(domain_loader[i]):
                avg_cost+= model.train(X,Y,i)

        avg_cost = avg_cost/(5*length_of_domain/batch_size)

        # print(avg_cost)



        if (epoch+1)%40==0:
            print('After {} epochs'.format(epoch))

            criterion = nn.MSELoss()
            feature_extractor = model.feature_extractor
            feature_extractor.eval()

            classifier = model.Classifier
            classifier.eval()

            with torch.no_grad():
                y_train = []
                train_out = []
                train_features_out = []
                for i in domain_loader:
                    for j, (X, Y) in enumerate(i):
                        feat = feature_extractor(X)
                        out = classifier(feat)
                        train_features_out.append(feat)
                        train_out.append(out)
                        y_train.append(Y)

                train_out = torch.cat(train_out).to(device)
                y_train = torch.cat(y_train).to(device)
                train_features_out = torch.cat(train_features_out).to(device)

                _, predicted = torch.max(train_out, dim=1)
                correct = sum(np.array((predicted == y_train).cpu()))
                accuracy = correct / len(y_train)
                print('Train Accuracy of the model: {} %'.format(accuracy))

            with torch.no_grad():

                predicted = []
                correct = 0
                for i in range(len(y_train)):
                    cost = []
                    for j in range(7):
                        cost.append(criterion(train_features_out[i], w2v[j]).item())

                    pred = cost.index(min(cost))
                    predicted.append(pred)
                    if pred == y_train[i]:
                        correct += 1

                accuracy = correct / len(y_train)
                print('MSE Train Accuracy of the model: {} %'.format(accuracy))

            with torch.no_grad():
                y_zs = []
                zs_out = []
                zs_features_out = []
                for j, (X, Y) in enumerate(zs_loader):
                    feat = feature_extractor(X)
                    zs_features_out.append(feat)
                    y_zs.append(Y)

                y_zs = torch.cat(y_zs).to(device)
                zs_features_out = torch.cat(zs_features_out).to(device)

                predicted = []
                correct = 0
                for i in range(len(y_zs)):
                    cost = []
                    for j in range(7):
                        cost.append(criterion(zs_features_out[i], w2v[j]).item())

                    pred = cost.index(min(cost))
                    predicted.append(pred)
                    if pred == y_zs[i]:
                        correct += 1

                accuracy = correct / len(y_zs)
                print('MSE Zero SHot Accuracy of the model: {} %'.format(accuracy))
                print(predicted)


            with torch.no_grad():

                test_tensor = torch.utils.data.TensorDataset(test_x, test_y)
                test_loader = torch.utils.data.DataLoader(dataset=test_tensor, batch_size=500, shuffle=False)

                y_test = []
                test_out = []
                test_features_out = []
                for i, (X, Y) in enumerate(test_loader):

                    feat = feature_extractor(X)
                    out = classifier(feat)
                    test_features_out.append(feat)
                    test_out.append(out)
                    y_test.append(Y)


                test_out = torch.cat(test_out).to(device)
                y_test = torch.cat(y_test).to(device)
                test_features_out = torch.cat(test_features_out).to(device)

                _, predicted = torch.max(test_out, dim=1)
                correct = sum(np.array((predicted == y_test).cpu()))
                print(correct)
                accuracy = correct / len(y_test)
                print('Test Accuracy of the model: {} %'.format(accuracy))

                predicted = []
                correct = 0
                for i in range(len(y_test)):
                    cost = []
                    for j in range(7):
                        cost.append(criterion(test_features_out[i], w2v[j]).item())

                    pred = cost.index(min(cost))
                    predicted.append(pred)
                    if pred == y_test[i]:
                        correct += 1

                accuracy = correct / len(y_test)
                print('MSE Test Accuracy of the model: {} %'.format(accuracy))
                PCA_TSNE(train_features_out.cpu(),y_train.cpu())


                PCA_TSNE(np.array(torch.cat((train_features_out, zs_features_out), 0).cpu()),(np.array(torch.cat((y_train, zs_y), 0).cpu())))

                del test_out

for dom in range(4):
    print("----------------------Domain.{}---------------------".format(dom))
    domain_specific_training(dom)
