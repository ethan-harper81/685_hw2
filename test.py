import torch
import random

x = torch.tensor([ i for i in range(20)])

random.shuffle(x)

num_positions = len(x)

d_model = 10

emb = torch.nn.Embedding(d_model, num_positions)

y = emb(x)

print(y)
print(y.shape)