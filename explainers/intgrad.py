import torch

from .explainer import *

class IntGradExplainer(Explainer):
  def __init__(self, is_batched, is_signed=False, num_steps=100, r2b_method=DontR2B()):
    super(IntGradExplainer, self).__init__(is_batched)
    self.is_signed = is_signed
    self.num_steps = num_steps
    self.r2b_method = r2b_method

  def get_explanation(self, model, x, w_start=None, w_final=None):
    assert len(model.out_shape) == 1 # Classification model
    if not self.is_batched:
      x = x.unsqueeze(0)

    y = model(x, w=torch.ones_like(x))
    target_class = y.argmax(dim=1)

    w_start = torch.zeros_like(x) if w_start is None else w_start
    w_final = torch.ones_like(x) if w_final is None else w_final
    intg = torch.zeros_like(w_start)
    num_steps = self.num_steps

    for k in range(num_steps):
      w_iter = w_start + (k/num_steps) * w_final
      w_iter.requires_grad_()
      y_iter = model(x, w=w_iter)
      logits = y_iter[range(len(target_class)), target_class]
      s = logits.sum()
      s.backward()
      intg += (w_final - w_start) * (w_iter.grad / num_steps)

    intg = intg if self.is_signed else intg.abs()
    b = self.r2b_method(intg)

    if not self.is_batched:
      return b[0]
    else:
      return b


