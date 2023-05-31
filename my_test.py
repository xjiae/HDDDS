import torch
import torch.nn as nn

from explainers import *
from models import *
from hddds import *


train_dataset = get_dataset('hai_sliding_train_100')
ret0 = train_dataset[0]
ret1 = train_dataset[1]
x0, y0, alpha0 = ret0['x'], ret0['y'], ret0['exp']
x1, y1, alpha1 = ret1['x'], ret1['y'], ret1['exp']
xx = torch.stack([x0, x1]).double()
# yy = torch.stack([y0, y1])
aa = torch.stack([alpha0, alpha1]).double()


lstm = SimpleLSTM(in_shape = (86,), out_shape = (1,)).double()

# breakpoint()
y = lstm(xx, aa)

# breakpoint()




# snet = SimpleNet(in_shape=(3,24,24), out_shape=(10,))
# x = torch.rand(7,*snet.in_shape)

# Select the top-100 bits to be hot
k = 10
# vgrad_exp = VGradExplainer(is_batched=True, r2b_method=TopKR2B(snet.in_shape, k))
vgrad_exp = VGradExplainer(is_batched=True, r2b_method=TopKR2B(lstm.in_shape, k))
ww = vgrad_exp.get_explanation(lstm, xx)
print(w.view(99*2,-1).sum(dim=1) == k)
breakpoint()


