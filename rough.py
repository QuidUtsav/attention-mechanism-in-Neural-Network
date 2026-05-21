import torch
import torch.nn as nn

layer = nn.Linear(8, 4)

x = torch.randn(2, 8)

y = layer(x)

print(x.shape)
print(y.shape)
print(layer.weight)