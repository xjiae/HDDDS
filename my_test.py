import torch
import torch.nn as nn

from explainers import *
from models import *
from hddds import *


snet = SimpleNet(in_shape=(3,24,24), out_shape=(10,))
x = torch.rand(7,*snet.in_shape)

# Select the top-100 bits to be hot
k = 100
vgrad_exp = VGradExplainer(is_batched=True, r2b_method=TopKR2B(snet.in_shape, k))
vgrad_w = vgrad_exp.get_explanation(snet, x)

igrad_exp = IntGradExplainer(is_batched=True, r2b_method=TopKR2B(snet.in_shape, k))
igrad_w = igrad_exp.get_explanation(snet, x)

