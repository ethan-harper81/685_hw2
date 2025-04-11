import torch
import random
import numpy as np

torch.manual_seed(0)

softmax = torch.nn.Softmax(dim = 1)

x = torch.rand([2,3])

print(x)

print(x.shape)

y = softmax(x)
y1 = torch.nn.functional.softmax(x, dim = 1)

z = torch.sum(y, dim = 1)
z1 = torch.sum(y1, dim = 1)

print(y)
print(z)

print(y1)
print(z1)