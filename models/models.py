import math
import torch
import torch.nn as nn


# A simple conv + feed forward network
class SimpleNet(nn.Module):
  def __init__(self, in_shape, out_shape, linear_dim=128):
    super(SimpleNet, self).__init__()
    self.in_shape = in_shape
    self.out_shape = out_shape
    self.in_dim = math.prod(list(in_shape))
    self.out_dim = math.prod(list(out_shape))
    self.linear_dim = linear_dim

    self.norm1 = nn.LayerNorm(self.in_dim)
    self.convs = nn.Sequential(
      nn.Conv1d(2, 64, 5, stride=1, padding=2),
      nn.ReLU(),
      nn.Conv1d(64, 32, 5, stride=1, padding=2),
      nn.Flatten(1),
      nn.AdaptiveAvgPool1d(linear_dim)
    )

    self.norm2 = nn.LayerNorm(linear_dim)
    self.linears = nn.Sequential(
      nn.Linear(linear_dim, linear_dim),
      nn.ReLU(),
      nn.Linear(linear_dim, linear_dim),
      nn.ReLU(),
      nn.Linear(linear_dim, self.out_dim)
    )

  def forward(self, x, w=None):
    w = torch.ones_like(x) if w is None else w
    assert x.shape == w.shape
    N = x.size(0) # N is batch dim
    x = self.norm1(x.view(N,1,self.in_dim))
    w = w.view(N,1,self.in_dim)
    z = torch.cat([x, w], dim=1)
    z = self.convs(z)   # (N,linear_dim)
    z = self.norm2(z)
    z = self.linears(z) # (N,out_dim)
    return z.view(N,*self.out_shape)


# A simple LSTM implementation
class SimpleLSTM(nn.Module):
  def __init__(self, in_shape, out_shape, hidden_dim=128):
    super(SimpleLSTM, self).__init__()
    self.in_shape = in_shape
    self.out_shape = out_shape
    self.in_dim = math.prod(list(in_shape))
    self.out_dim = math.prod(list(out_shape))
    self.hidden_dim = hidden_dim

    self.lstm1 = nn.LSTM(input_size=2*self.in_dim, hidden_size=hidden_dim, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
    self.lstm3 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
    self.linear = nn.Linear(hidden_dim, self.out_dim)

  def forward(self, x, w=None):
    w = torch.ones_like(x) if w is None else w
    assert x.shape == w.shape
    assert x.shape[2:] == self.in_shape
    N, L = x.shape[0:2]
    z = torch.cat([x.flatten(2), w.flatten(2)], dim=2).type(torch.DoubleTensor).cuda() # (N,L,2*d)
    # breakpoint()
    z, _ = self.lstm1(z)
    z, _ = self.lstm2(z)
    z, _ = self.lstm3(z)
    z = self.linear(z)
    return z.view(N,L,*self.out_shape)

#

