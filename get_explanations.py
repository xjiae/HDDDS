import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch.utils.data as tud

from dataset import *
from models import *

import my_openxai

# Some configurations
class ExplainerConfigs:
  def __init__(self):
    pass

# Gradients
class GradConfigs(ExplainerConfigs):
  def __init__(self, abs_value=False, train_mode=False):
    super(GradConfigs, self).__init__() 
    self.abs_value = abs_value
    self.train_mode = train_mode

  def desc_str(self):
    return "gradu" if self.abs_value else "grads"

# Integrated gradients
class IntGradConfigs(ExplainerConfigs):
  def __init__(self, baseline=None, train_mode=False):
    super(IntGradConfigs, self).__init__()
    self.baseline = baseline
    self.train_mode = train_mode

  def desc_str(self):
    return "intg"

# LIME
class LimeConfigs(ExplainerConfigs):
  def __init__(self, x_train, num_samples=100):
    super(LimeConfigs, self).__init__()
    self.param_dict_lime = {
      "dataset_tensor" : x_train,
      "kernel_width" : 0.75,
      "std" : float(np.sqrt(0.05)),
      "mode" : "tabular",
      "sample_around_instance" : False,
      "n_samples" : num_samples,
      "discretize_continuous" : False
    }
    self.train_mode = False

  def desc_str(self):
    return "lime"

# SHAP
class ShapConfigs(ExplainerConfigs):
  def __init__(self, num_samples=100):
    super(ShapConfigs, self).__init__()
    self.param_dict_shap = {
      "subset_size" : num_samples
    }
    self.train_mode = False

  def desc_str(self):
    return "shap"


# The stuff for MVTec
def get_mvtec_explanation(model, dataset, configs,
                          custom_desc = None,
                          misc_data = None,
                          num_todo = None,
                          post_process_fun = None,
                          save_every_k = 20,
                          device = "cuda",
                          do_save = True,
                          saveto = None,
                          seed = 1234):
  assert isinstance(model, XwModel)
  if do_save: assert saveto is not None
  num_todo = len(dataset) if num_todo is None else num_todo
  torch.manual_seed(1234)
  perm = torch.randperm(len(dataset))
  dataset = tud.Subset(dataset, indices=perm)

  if isinstance(configs, GradConfigs):
    explainer = my_openxai.Explainer(method="grad", model=model)
  elif isinstance(configs, IntGradConfigs):
    explainer = my_openxai.Explainer(method="ig", model=model, dataset_tensor=configs.baseline)
  elif isinstance(configs, LimeConfigs):
    explainer = my_openxai.Explainer(method="lime", model=model, param_dict_lime=configs.param_dict_lime)
  elif isinstance(configs, ShapConfigs):
    explainer = my_openxai.Explainer(method="shap", model=model, param_dict_shap=configs.param_dict_shap)
  else:
    raise NotImplementedError()

  model.eval().to(device)

  all_xs, all_ys, all_ws, all_w_exps = [], [], [], []
  pbar = tqdm(range(num_todo))
  for i in pbar:
    x, y, w = dataset[i]
    xx, yy, w = x.unsqueeze(0).to(device), torch.tensor([y]).to(device), w.to(device)

    # ww_exp : (3,256,256), need to compress it to (1,256,256)
    ww_exp = explainer.get_explanation(xx, yy).view(x.shape)
    if callable(post_process_fun):
      w_exp = post_process_fun(ww_exp)
    else:
      w_exp = ww_exp.max(dim=0).values.clamp(0,1).view(w.shape).float()

    all_xs.append(x.cpu())
    all_ys.append(y)
    all_ws.append(w.cpu())
    all_w_exps.append(w_exp.cpu())

    # Speedhack: save once every few iters, or if we're near the end
    if do_save and (i % save_every_k == 0 or len(pbar) - i < 2):
      model_class = model.__class__
      state_dict = model.state_dict()
      stuff = {
          "dataset" : dataset,
          "model_class" : model_class,
          "model_state_dict" : state_dict,
          "method" : configs.desc_str(),
          "num_total" : len(all_xs),
          "w_exps" : all_w_exps,
          "ws" : all_ws,
          "custom_desc" : custom_desc,
          "misc_data" : misc_data,
      }
      torch.save(stuff, saveto)

  return stuff

# The stuff for tabular
def get_tabular_explanation(model, dataset, configs,
                          custom_desc = None,
                          misc_data = None,
                          num_todo = None,
                          post_process_fun = None,
                          save_every_k = 20,
                          device = "cuda",
                          do_save = True,
                          saveto = None,
                          seed = 1234):
  assert isinstance(model, XwModel)
  if do_save: assert saveto is not None
  num_todo = len(dataset) if num_todo is None else num_todo
  _, indices = train_test_split(range(len(dataset)), test_size=num_todo, stratify=dataset.y, random_state=seed)
  dataset = torch.utils.data.Subset(dataset, indices)


  if isinstance(configs, GradConfigs):
    explainer = my_openxai.Explainer(method="grad", model=model)
  elif isinstance(configs, IntGradConfigs):
    explainer = my_openxai.Explainer(method="ig", model=model, dataset_tensor=configs.baseline)
  elif isinstance(configs, LimeConfigs):
    explainer = my_openxai.Explainer(method="lime", model=model, param_dict_lime=configs.param_dict_lime)
  elif isinstance(configs, ShapConfigs):
    explainer = my_openxai.Explainer(method="shap", model=model, param_dict_shap=configs.param_dict_shap)
  else:
    raise NotImplementedError()

 

  all_xs, all_ys, all_ws, all_w_exps = [], [], [], []
  pbar = tqdm(range(num_todo))

  for i in pbar:
    x, y, w, l = dataset[i]
    xx, yy, w = x.unsqueeze(0).to(device).contiguous(), torch.tensor([y]).to(device), w.to(device)


    if configs.train_mode:
      model.train().to(device)
      ww_exp = explainer.get_explanation(xx, yy, configs.train_mode).view(x.shape)
      model.eval().to(device)
    else:
      ww_exp = explainer.get_explanation(xx, yy).view(x.shape)
    
    if callable(post_process_fun):
      w_exp = post_process_fun(ww_exp)
    else:
      w_exp = ww_exp.clamp(0,1).view(w.shape).float()

    all_xs.append(x.cpu())
    all_ys.append(y)
    all_ws.append(w.cpu())
    all_w_exps.append(w_exp.cpu())

    # Speedhack: save once every few iters, or if we're near the end
    if do_save and (i % save_every_k == 0 or len(pbar) - i < 2):
      model_class = model.__class__
      state_dict = model.state_dict()
      stuff = {
          "dataset" : dataset,
          "model_class" : model_class,
          "model_state_dict" : state_dict,
          "method" : configs.desc_str(),
          "num_total" : len(all_xs),
          "w_exps" : all_w_exps,
          "ws" : all_ws,
          "custom_desc" : custom_desc,
          "misc_data" : misc_data,
      }
      torch.save(stuff, saveto)

  return stuff

