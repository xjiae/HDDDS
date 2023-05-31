import torch
import torch.nn as nn

from .explainer import *

class VGradExplainer(Explainer):
  def __init__(self, is_batched, is_signed=False, r2b_method=DontR2B()):
    super(VGradExplainer, self).__init__(is_batched)
    self.is_signed = is_signed
    self.r2b_method = r2b_method

  def get_explanation(self, model, x):
    assert isinstance(model, nn.Module)
    # assert len(model.out_shape) == 1 # Classification model
    
    if not self.is_batched:
      x = x.unsqueeze(0)

    # Compute the max stuff
    w = torch.ones_like(x)
    w.requires_grad_()
    y = model(x, w=w)
    v, _ = y.max(dim=1)

    # Take gradients
    for vi in v:
      vi.backward(retain_graph=True)

    grad = w.grad if self.is_signed else w.grad.abs()
    b = self.r2b_method(grad)

    if not self.is_batched:
      return b[0] # This undoes the batch stuff we did
    else:
      return b

