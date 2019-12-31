import torch
device = 'cuda:2'
seed = 997


words = []
with open('wiki2vec_words.txt', 'rb') as f:
    for l in f:
        l = l.decode().split()
        words.append(l[0])


vectors = torch.load('wiki2vec_vectors.txt')
print('WIKI2VEC 100*')
print(vectors.shape)
classes = ['dog','elephant', 'giraffe','guitar', 'horse', 'house', 'person']
w2v = []
for i in classes:
    idx = words.index(i)
    w2v.append(vectors[idx])
w2v = torch.stack(w2v).to(device).float()


# To Normalize Uncomment the following code
# for i in range(7):
#     w2v[i] = (w2v[i] - torch.min(w2v[i])) / (torch.max(w2v[i]) - torch.min(w2v[i]))
