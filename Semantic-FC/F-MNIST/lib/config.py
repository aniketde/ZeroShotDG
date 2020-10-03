import torch
device = 'cpu'
seed = 107


words = []
with open('fmnist_.txt', 'rb') as f:
    for l in f:
        l = l.decode().split()
        words.append(l[0])
vectors = torch.load('fmnist_vectors.txt')

classes = {'Tshirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot'}
w2v = []
for i in classes:
    idx = words.index(i)
    w2v.append(vectors[idx])
w2v = torch.stack(w2v).to(device).float()
