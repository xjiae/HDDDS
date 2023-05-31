import torch
import torch.nn as nn

from explainers import *
from models import *

snet = SimpleNet(in_shape=(3,24,24), out_shape=(10,))
x = torch.rand(7,*snet.in_shape)

# Select the top-100 bits to be hot
k = 100
vgrad_exp = VGradExplainer(is_batched=True, r2b_method=TopKR2B(snet.in_shape, k))
w = vgrad_exp.get_explanation(snet, x)
print(w.view(7,-1).sum(dim=1) == k)


