import math
import torch
from abc import ABC, abstractmethod


# Generic way to convert a continuous-valued w to a binary one
class Real2BinaryMethod:
  def __init__(self, shape):
    self.shape = shape
    self.dim = math.prod(list(shape))
  
  def to_binary(self, w):
    raise NotImplementedError()

  def __call__(self, *args, **kwargs):
    return self.to_binary(*args, **kwargs)

# Take the top k
class TopKR2B(Real2BinaryMethod):
  def __init__(self, shape, k):
    super(TopKR2B, self).__init__(shape)
    self.k = k
    assert k >= 1

  def to_binary(self, w):
    assert w.shape[2:] == self.shape
    w = w.view(-1, self.dim)
    hot_inds = w.sort(dim=1, descending=True).indices[:,:self.k]
    b = torch.zeros_like(w)
    for i, hi in enumerate(hot_inds):
      b[i,hi] = 1.0
    return b.view(-1,*self.shape)

# Take a bunch at random
class RandR2B(Real2BinaryMethod):
  def __init__(self, shape, hot_prob):
    super(RandR2B, self).__init__(shape)
    self.hot_prob = hot_prob

  def to_binary(self, w):
    assert w.shape[1:] == self.shape
    b = torch.rand(w.shape)
    return (b <= self.hot_prob) * 1.0

# Don't do it; just return w as-is
class DontR2B(Real2BinaryMethod):
  def __init__(self):
    super(DontR2B, self).__init__((0,))

  def to_binary(self, w):
    return w


# Generic explanation thing
class Explainer:
  def __init__(self, is_batched):
    self.is_batched = is_batched

  # Find the w
  def get_explanation(self, model, x):
    raise NotImplementedError()


'''
class Explainer(ABC):
    """
    Abstract class to implement custom explanation methods for a given.
    Parameters
    ----------
    model: any model implemented in pytorch
        Classifier we wish to explain.
    Methods
    -------
    get_explanations:
        Generate explanations for given input.
    Returns
    -------
    None
    """

    def __init__(self):
        # self.model = model
        pass

    # @abstractmethod
    # def get_explanation(self, model: torch.nn, inputs: torch.tensor):
    #     """
    #     Generate explanations for given input/s.
    #     Parameters
    #     ----------
    #     inputs: torch.tensor
    #         Input in two-dimensional shape (m, n).
    #     label: torch.tensor
    #         Label
    #     Returns
    #     -------
    #     torch.tensor
    #         Explanation vector/matrix.
    #     """
    #     pass
'''
    
