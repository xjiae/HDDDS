#
import math
import torch
import torch.nn as nn
import shap

import copy



# Wrap something around a MuS model so that we can exploit CUDA better
class Wrapper(nn.Module):
  def __init__(self, model, x):
    super(Wrapper, self).__init__()
    self.model = model.cuda()
    assert isinstance(x, torch.Tensor)
    self.x = x.cuda()
  
  def forward(self, alpha):
    torch.cuda.empty_cache()
    N, p = alpha.shape
    device = alpha.device
    # xx = torch.stack(N*[self.x])

    # y = self.model(xx, alpha=alpha.cuda())
    y = self.model(self.x)

    return y.to(device)