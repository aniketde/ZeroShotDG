import torch
device = 'cpu'
seed = 107

words = []
with open('cifar_words.txt', 'rb') as f:
    for l in f:
        l = l.decode().split()
        words.append(l[0])
vectors = torch.load('cifar_vectors.txt')

classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
w2v = []
for i in classes:
    idx = words.index(i)
    w2v.append(vectors[idx])
w2v = torch.stack(w2v).to(device).float()

