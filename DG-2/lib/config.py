import torch
device = 'cuda:2'
seed = 999


words = []
with open('words.txt', 'rb') as f:
    for l in f:
        l = l.decode().split()
        words.append(l[0])
vectors = torch.load('vectors.txt')

classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
w2v = []
for i in classes:
    idx = words.index(i)
    w2v.append(vectors[idx])
w2v = torch.stack(w2v).to(device).float()

for i in range(10):
    w2v[i] = (w2v[i] - torch.min(w2v[i])) / (torch.max(w2v[i]) - torch.min(w2v[i]))





# import scipy.io
# mat = scipy.io.loadmat('wordTable.mat')
# classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# vectors = mat['wordTable']
# vectors = np.array(vectors)
# w2v = torch.tensor(np.transpose(vectors)).to(device).float()
