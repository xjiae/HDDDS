import math
import torch
import torch.nn as nn
import torchvision

from .fastflow import *


# The two-argument model
class XwModel(nn.Module):
  def __init__(self, in_shape, out_shape, w_shape=None):
    super(XwModel, self).__init__()
    self.in_shape = in_shape
    self.out_shape = out_shape
    self.w_shape = in_shape if w_shape is None else w_shape
    self.in_dim = math.prod(list(in_shape))
    self.out_dim = math.prod(list(out_shape))
    self.w_dim = math.prod(list(w_shape))

  def forward(self, x, w):
    raise NotImplementedError()


# A simple conv + feed forward network
class SimpleNet(XwModel):
  def __init__(self, in_shape, out_shape, linear_dim=128, auto_reshape=True):
    super(SimpleNet, self).__init__(in_shape, out_shape, w_shape=in_shape)
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

    self.auto_reshape = auto_reshape

  def forward(self, x, w=None):
    N = x.size(0) # N is batch dim
    w = torch.ones(N, *self.w_shape).to(x.device) if w is None else w
    # Reshape as necessary
    if self.auto_reshape:
      x = x.view(N, *self.in_shape)
      w = w.view(N, *self.w_shape)

    x = self.norm1(x.view(N,1,self.in_dim))
    w = w.view(N,1,self.in_dim)
    z = torch.cat([x, w], dim=1)
    z = self.convs(z)   # (N,linear_dim)
    z = self.norm2(z)
    z = self.linears(z) # (N,out_dim)

    # If it's classification we do a normalization
    if len(self.out_shape) == 1:
      return z.softmax(dim=1)
    else:
      return z.view(N,*self.out_shape)


# A simple LSTM implementation
class SimpleLSTM(XwModel):
  def __init__(self, in_shape, out_shape, hidden_dim=128, auto_reshape=True, return_mode="last"):
    super(SimpleLSTM, self).__init__(in_shape, out_shape, w_shape=in_shape)
    self.hidden_dim = hidden_dim
    self.lstm1 = nn.LSTM(input_size=2*self.in_dim, hidden_size=hidden_dim, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
    self.lstm3 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
    self.linear = nn.Linear(hidden_dim, self.out_dim)
    self.auto_reshape = auto_reshape
    self.return_mode = return_mode

  def forward(self, x, w=None):
    N = x.size(0) # N is batch dim
    L = x.numel() // (N * self.in_dim) # Horizon
    w = torch.ones(N, L, *self.w_shape).to(x.device) if w is None else w
    if self.auto_reshape:
      x = x.view(N, L, *self.in_shape)
      w = w.view(N, L, *self.w_shape)

    z = torch.cat([x.flatten(2), w.flatten(2)], dim=2) # (N,L,2*d)
    z = z.type(torch.FloatTensor) # Force a convert because nn.LSTM is a coward
    z, _ = self.lstm1(z)
    z, _ = self.lstm2(z)
    z, _ = self.lstm3(z)
    z = self.linear(z)
    z = z.view(N,L,*self.out_shape)

    if self.return_mode == "last":
      return z[:,-1]
    elif self.return_mode == "all":
      return z
    elif self.return_mode == "prob":
      ll = z[:, -1]
      rr = 1 - z[:, -1]
      return torch.hstack((ll, rr))
    else:
      raise NotImplementedError()


# x --[fastflow]--> (x, hmap, w) --[resnet]--> [-1,+1]
class MyFastResA(XwModel):
  def __init__(self,
               in_shape = (3,256,256),
               out_shape = (1,),
               w_shape = (1,256,256),
               backbone_name = "resnet18",
               flow_steps = 8,
               conv3x3_only = False,
               freeze_fastflow = False,
               auto_reshape = True,
               return_mode = "scalar_score"):
      super(MyFastResA, self).__init__(in_shape, out_shape, w_shape=w_shape)
      self.backbone_name = backbone_name
      self.ffmodel = FastFlow(backbone_name, flow_steps, in_shape[1], conv3x3_only)

      assert len(in_shape) == 3 and len(w_shape) == 3
      self.input_channels = in_shape[0]

      # Freeze fast flow weights
      if freeze_fastflow:
        for param in self.ffmodel.parameters():
          param.requires_grad = False

      resnet = torchvision.models.resnet18()
      resnet.conv1 = nn.Conv2d(self.input_channels+2, 64, kernel_size=7, stride=2, padding=3)
      self.resnet = resnet
      self.hmap_shape = w_shape
      self.auto_reshape = auto_reshape
      self.return_mode = return_mode

  def forward(self, x, w=None):
    N = x.size(0) # N is batch dim
    w = torch.ones(N, *self.w_shape).to(x.device) if w is None else w
    # Reshape as necessary
    if self.auto_reshape:
      x = x.view(N, *self.in_shape)
      w = w.view(N, *self.w_shape)

    ret = self.ffmodel(x)
    hmap = ret["anomaly_map"]
    z = self.resnet(torch.cat([x, hmap, w], dim=1))
    z = torch.tanh(z[:,0]).view(-1,1)
    ret['anomaly_score'] = z

    if self.return_mode == "scalar_score":
      return z  # (N,1)
    elif self.return_mode == "two_class":
      anom_prob = (z + 1) / 2
      good_prob = 1 - anom_prob
      return torch.cat([good_prob, anom_prob]) # (N,2)
    else:
      return ret


# Use this to explain one single x (no batch!)
class XWrapper(nn.Module):
  def __init__(self, model, x, auto_reshape=True):
    super(XWrapper, self).__init__()
    # assert isinstance(model, XwModel)
    # assert model.in_shape == x.shape
    self.model = model
    self.x = x
    self.auto_reshape = auto_reshape
  
  def forward(self, w):
    N = w.size(0)
    w = w.view(N, *self.x.shape) if self.auto_reshape else w
    x = torch.stack(N * [self.x])
    y = self.model(x, w)
    # Check if the probablities sum to 1
    if y.ndim == 2 and not torch.allclose(y.sum(dim=1), torch.ones(N).to(w.device)):
      return y.softmax(dim=1)
    else:
      return y


