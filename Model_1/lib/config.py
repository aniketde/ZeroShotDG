import torch
device = 'cpu'
seed = 999

words = []

with open('words.txt', 'rb') as f:
    for l in f:
        l = l.decode().split()
        words.append(l[0])
vectors = torch.load('vectors.txt')


clothes = ['Tshirt', 'trouser', 'Pullover', 'dress', 'Coat', 'SANDAL', 'Shirt', 'SNEAKER', 'Bag', 'BOOT']
w2v = []
for i in clothes:
    idx = words.index(i)
    w2v.append(vectors[idx])
w2v = torch.stack(w2v).to(device).float()

for i in range(10):
    w2v[i] = (w2v[i] - torch.min(w2v[i])) / (torch.max(w2v[i]) - torch.min(w2v[i]))
