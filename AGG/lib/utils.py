
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from lib.config import *
import matplotlib
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
    print(output.shape)
    print(w2v.shape)
    output = np.append(output, np.array(w2v.cpu()),axis = 0)
    y = np.append(y, [7, 8,9,10,11,12,13])

    print(y)
    pca_model = PCA()
    X_embedded = pca_model.fit_transform(output)
    print('Variance Ratio:', pca_model.explained_variance_ratio_[:2])


    ix1 = X_embedded[:, 0]
    ix2 = X_embedded[:, 1]
    cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
            'tab:olive', 'tab:cyan']
    classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    fig, ax = plt.subplots()
    print(y)
    for g in np.unique(y):
        if g<7:
            ix = np.where(y == g)
            ax.scatter(ix1[ix],ix2[ix],c=cmap[g], label=classes[g],s=1)
        else:
            ix = np.where(y == g)
            ax.scatter(ix1[ix],ix2[ix],c=cmap[g%7], marker='+', label=classes[g-7],s=100)

    ax.legend()
    plt.show()



    X_embedded = TSNE(n_components=2).fit_transform(output)
    ix1 = X_embedded[:, 0]
    ix2 = X_embedded[:, 1]

    fig, ax = plt.subplots()
    for g in np.unique(y):
        print(g)
        if g<7:
            ix = np.where(y == g)
            ax.scatter(ix1[ix],ix2[ix],c=cmap[g], label=classes[g],s=1)
        else:
            ix = np.where(y == g)
            ax.scatter(ix1[ix],ix2[ix],c=cmap[(g)%7], marker='+',s=500
                       )
    ax.legend(loc='best')
    # plt.show()
    # , label = 'WordVector' + classes[g - 7],
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    #
    # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()