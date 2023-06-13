import copy
import math
import torch
import torch.nn as nn
import torchvision

import transformers
from transformers import (
  AutoTokenizer,
  AutoModelForQuestionAnswering,
)

from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .fastflow import *


class ShapedModel(nn.Module):
  def __init__(self, in_shape, out_shape):
    super(ShapedModel, self).__init__()
    self.in_shape = in_shape
    self.out_shape = out_shape
    self.in_dim = math.prod(list(in_shape))
    self.out_dim = math.prod(list(out_shape))


class MySimpleNet(ShapedModel):
  def __init__(self, in_shape, out_shape, linear_dim=128, softmax_output=True):
    super(MySimpleNet, self).__init__(in_shape, out_shape)
    self.linear_dim = linear_dim
    self.softmax_output = softmax_output
    self.norm1 = nn.LayerNorm(self.in_dim)
    self.convs = nn.Sequential(
      nn.Conv1d(1, 64, 5, stride=1, padding=2),
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

  def forward(self, x):
    N = x.size(0)
    x = x.view(N, *self.in_shape)

    z = x.view(N,1,self.in_dim)
    z = self.norm1(z)
    z = self.convs(z)   # (n, linear_dim)
    z = self.norm2(z)
    z = self.linears(z) # (N,out_dim)

    if self.softmax_output:
      z = torch.sigmoid(z) if z.size(1) == 1 else z.softmax(dim=1)

    return z.view(N,*self.out_shape)


# Logistic regression
class MyLogisticRegression(ShapedModel):
  def __init__(self, in_shape):
    super(MyLogisticRegression, self).__init__(in_shape, (2,))
    self.linear = torch.nn.Linear(self.in_dim, 2)

  def forward(self, x):
    z = x.flatten(1).float()
    z = self.linear(z)
    y = torch.softmax(z, dim=1) # 0: good prob, 1: anom prob
    return y
   

# x --[fastflow]--> (x, hmap, w) --[resnet]--> (N,2)
class MyFastResA(ShapedModel):
  def __init__(self,
               in_shape = (3,256,256),
               backbone_name = "resnet18",
               flow_steps = 8,
               conv3x3_only = False,
               freeze_fastflow = True,
               return_mode = "two_class"):
    super(MyFastResA, self).__init__(in_shape, (2,))
    self.backbone_name = backbone_name
    self.ffmodel = FastFlow(backbone_name, flow_steps, in_shape[1], conv3x3_only)

    assert len(in_shape) == 3
    self.input_channels = in_shape[0]

    # Freeze fast flow weights
    if freeze_fastflow:
      for param in self.ffmodel.parameters():
        param.requires_grad = False

    resnet = torchvision.models.resnet18()
    resnet.conv1 = nn.Conv2d(self.input_channels+1, 64, kernel_size=7, stride=2, padding=3)
    self.resnet = resnet
    self.return_mode = return_mode

  def forward(self, x):
    N = x.size(0) # N is batch dim
    x = x.view(N, *self.in_shape)
    ret = self.ffmodel(x)
    hmap = ret["anomaly_map"]
    z = self.resnet(torch.cat([x, hmap], dim=1))
    z = z[:,0:2]
    z = torch.softmax(z, dim=1)
    ret['anomaly_score'] = z[:,1]
    if self.return_mode == "two_class":
      return z
    else:
      return ret

  
# A simple LSTM implementation for binary classification
class MyLSTM(ShapedModel):
  def __init__(self, in_shape,
               hidden_dim = 128,
               auto_reshape = True,
               return_mode = "last"):
    super(MyLSTM, self).__init__(in_shape, (2,))
    self.hidden_dim = hidden_dim
    self.lstm1 = nn.LSTM(input_size=self.in_dim, hidden_size=hidden_dim, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
    self.lstm3 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
    self.linear = nn.Linear(hidden_dim, self.out_dim)
    self.return_mode = return_mode

  def forward(self, x):
    N = x.size(0) # N is batch dim
    L = x.numel() // (N * self.in_dim) # Horizon

    z = x.view(N,L,self.in_dim).float()
    z, _ = self.lstm1(z)
    z, _ = self.lstm2(z)
    z, _ = self.lstm3(z)
    z = self.linear(z)  # (N,L,2)
    z = z.softmax(dim=2)

    if self.return_mode == "last":
      return z[:,-1]  # (N,2)
    elif self.return_mode == "all":
      return z    # (N,L,2)
    elif self.return_mode == "exists":
      anom_probs = z[:,:,1]  # (N,L)
      max_probs = anom_probs.max(dim=1).values.view(N,1)
      return torch.cat([1-max_probs, max_probs], dim=1)  # (N,2)
    else:
      raise NotImplementedError()


#
class MySquad(nn.Module):
  def __init__(self,
               name_or_model,
               tokenizer,
               embed_fn = None,
               input_mode = "dict",
               return_mode = "all",
               embeds_dim = 768,
               pretrained_kwargs_dict= {}):
    super(MySquad, self).__init__()
    if isinstance(name_or_model, str):
      self.model = AutoModelForQuestionAnswering.from_pretrained(name_or_model, **pretrained_kwargs_dict)
    else:
      assert isinstance(name_or_model, nn.Module)
      self.model = copy.deepcopy(name_or_model)

    assert input_mode in ["inputs_embeds", "input_ids", "dict"]
    self.input_mode = input_mode
    self.return_mode = return_mode
    self.embeds_dim = embeds_dim
    self.embed_fn = self.model.get_input_embeddings() if embed_fn is None else embed_fn
    self.mask_token_id = tokenizer.mask_token_id
    self.mask_token_pt = self.embed_fn(torch.tensor(self.mask_token_id)).detach()

  def forward(self, x=None, w=None, **kwargs):
    # Logical implies
    if self.input_mode == "input_ids":
      x = self.embed_fn(x)
    elif self.input_mode == "dict":
      if "input_ids" in kwargs.keys():
        x = kwargs["input_ids"]
        x = self.embed_fn(x)
        # Delete since we've extracted embedding; otherwise model complains
        del kwargs["input_ids"]
      else:
        assert "inputs_embeds" in kwargs.keys()
        x = kwargs["inputs_embeds"]
    else:
      assert self.input_mode == "inputs_embeds"
      x = x.view(x.size(0),-1,self.embeds_dim)

    N, L, _ = x.shape # (batch_size, seq_len, embed_dim)
    w = torch.ones(N,L).to(x.device) if w is None else w
    w = w.view(N,L,1)

    # Now combine x and w
    # mask_token_pt = self.embed_fn(torch.tensor(self.mask_token_id))
    mask_token_pt = self.mask_token_pt.to(x.device)
    x_noised = w * x + (1 - w) * mask_token_pt
    outputs = self.model(inputs_embeds=x_noised, **kwargs)
    assert isinstance(outputs, QuestionAnsweringModelOutput)

    if self.return_mode == "all":
      return outputs
    elif self.return_mode == "start_logits":
      return torch.softmax(outputs.start_logits, dim=1)
    elif self.return_mode == "end_logits":
      return torch.softmax(outputs.end_logits, dim=1)
    else:
      raise NotImplementedError()


