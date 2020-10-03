
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=256, H=8, W=8):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def PCA_TSNE(output, y):
    output = np.append(output, np.array(w2v.cpu()), axis=0)
    y = np.append(y, [7, 8, 9, 10, 11, 12, 13])

    pca_model = PCA()
    X_embedded = pca_model.fit_transform(output)
    print('Variance Ratio:', pca_model.explained_variance_ratio_[:2])


    ix1 = X_embedded[:, 0]
    ix2 = X_embedded[:, 1]
    cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
            'tab:olive', 'tab:cyan']
    classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fig, ax = plt.subplots()
    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(ix1[ix],ix2[ix],c=cmap[g], label=classes[g],s=1)
    ax.legend()
    plt.show()



    X_embedded = TSNE(n_components=2).fit_transform(output)
    ix1 = X_embedded[:, 0]
    ix2 = X_embedded[:, 1]

    fig, ax = plt.subplots()
    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(ix1[ix],ix2[ix], label=classes[g],s=1)
    ax.legend()
    plt.show()
