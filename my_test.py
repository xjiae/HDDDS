import torch
import torch.nn as nn
import numpy as np

from my_openxai import Explainer

# from explainers import *
from models import *
from hddds import *


snet = SimpleNet(in_shape=(3,24,24), out_shape=(10,))
x = torch.rand(7,*snet.in_shape)

# Select the top-100 bits to be hot
# k = 100
# vgrad_exp = VGradExplainer(is_batched=True, r2b_method=TopKR2B(snet.in_shape, k))
# vgrad_w = vgrad_exp.get_explanation(snet, x)

# igrad_exp = IntGradExplainer(is_batched=True, r2b_method=TopKR2B(snet.in_shape, k))
# igrad_w = igrad_exp.get_explanation(snet, x)

test_lin = nn.Sequential(
    nn.Linear(in_features=64, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=8),
    nn.Softmax(dim=1))

class LimeWrapper:
  def __init__(self, model):
    self.model = model

  def predict(self, x_np):
    # print(f"x_np: {x_np.shape}")
    # return self.model(torch.tensor(x_np))
    return self.model(torch.tensor(np.float32(x_np)))



grad_explainer = Explainer(method="grad", model=XWrapper(snet, torch.rand(3,24,24)))

lime_w_data = torch.rand(100,3,24,24)
lime_explainer = Explainer(method="lime", model=XWrapper(snet, torch.rand(3,24,24)), dataset_tensor=lime_w_data)


n_test = 5
w_test = torch.rand(n_test, 3,24,24)
lbl_test = torch.randint(0,10, (n_test,))

grad_alpha = grad_explainer.get_explanation(w_test, lbl_test)
lime_alpha = lime_explainer.get_explanation(w_test, lbl_test)




