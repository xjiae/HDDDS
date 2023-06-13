import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch.utils.data as tud

from datasets import *
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
      "sample_around_instance" : True,
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
def get_mvtec_explanations(model, dataset, configs,
                           custom_desc = None,
                           misc_data = None,
                           num_todo = None,
                           post_process_fun = None,
                           save_every_k = 20,
                           device = "cuda",
                           do_save = True,
                           saveto = None,
                           save_small = True,
                           seed = 1234):
  model.eval().to(device)

  if do_save: assert saveto is not None

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

  torch.manual_seed(seed)
  num_todo = len(dataset) if num_todo is None else num_todo
  perm = torch.randperm(len(dataset))
  todo_indices = perm[:num_todo]
  pbar = tqdm(todo_indices)

  all_xs, all_ys, all_ws, all_w_exps = [], [], [], []
  for i, todo_ind in enumerate(pbar):
    desc_str = f"mvtec {configs.desc_str()}"
    pbar.set_description(desc_str)

    x, y, w = dataset[todo_ind]
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
          "model_class" : str(model_class),
          "model_state_dict" : None if save_small else state_dict,
          "method" : configs.desc_str(),
          "num_total" : len(all_xs),
          "w_exps" : all_w_exps,
          "ws" : all_ws,
          "todo_indices" : todo_indices,
          "custom_desc" : custom_desc,
          "misc_data" : misc_data,
      }
      torch.save(stuff, saveto)


  return stuff

# The stuff for tabular
def get_timeseries_explanations(model, dataset, configs,
                                custom_desc = None,
                                misc_data = None,
                                num_todo = None,
                                post_process_fun = None,
                                save_every_k = 20,
                                device = "cuda",
                                do_save = True,
                                saveto = None,
                                save_small = True,
                                seed = 1234):
  model.eval().to(device)
  if do_save: assert saveto is not None

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

  torch.manual_seed(seed)
  num_todo = len(dataset) if num_todo is None else num_todo
  perm = torch.randperm(len(dataset))
  todo_indices = perm[:num_todo]
  pbar = tqdm(todo_indices)

  all_xs, all_ys, all_ws, all_w_exps = [], [], [], []
  for i, todo_ind in enumerate(pbar):
    desc_str = f"timeseries (in_shape {model.in_shape}) {configs.desc_str()}"
    pbar.set_description(desc_str)
    x, y, w = dataset[todo_ind]
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
          "model_state_dict" : None if save_small else state_dict,
          "method" : configs.desc_str(),
          "num_total" : len(all_xs),
          "w_exps" : all_w_exps,
          "ws" : all_ws,
          "todo_indices" : todo_indices,
          "custom_desc" : custom_desc,
          "misc_data" : misc_data,
      }
      torch.save(stuff, saveto)

  return stuff


# The stuff for squad
def get_squad_explanations(model, dataset, configs,
                           custom_desc = None,
                           misc_data = None,
                           num_todo = None,
                           post_process_fun = None,
                           save_every_k = 20,
                           device = "cuda",
                           do_save = True,
                           saveto = None,
                           save_small = True,
                           seed = 1234):
  if do_save: assert saveto is not None

  # Wrap so that it now directly takes inputs_embeds
  amodel = copy.deepcopy(model).eval().to(device)
  bmodel = copy.deepcopy(model).eval().to(device)
  amodel.return_mode = "start_logits"
  bmodel.return_mode = "end_logits"

  embed_fn = amodel.model.get_input_embeddings()

  if isinstance(configs, GradConfigs):
    aexplainer = my_openxai.Explainer(method="grad", model=amodel)
    bexplainer = my_openxai.Explainer(method="grad", model=bmodel)
  elif isinstance(configs, IntGradConfigs):
    aexplainer = my_openxai.Explainer(method="ig", model=amodel, dataset_tensor=configs.baseline)
    bexplainer = my_openxai.Explainer(method="ig", model=bmodel, dataset_tensor=configs.baseline)
  elif isinstance(configs, LimeConfigs):
    aexplainer = my_openxai.Explainer(method="lime", model=amodel, param_dict_lime=configs.param_dict_lime)
    bexplainer = my_openxai.Explainer(method="lime", model=bmodel, param_dict_lime=configs.param_dict_lime)
  elif isinstance(configs, ShapConfigs):
    aexplainer = my_openxai.Explainer(method="shap", model=amodel, param_dict_shap=configs.param_dict_shap)
    bexplainer = my_openxai.Explainer(method="shap", model=bmodel, param_dict_shap=configs.param_dict_shap)
  else:
    raise NotImplementedError()

  torch.manual_seed(seed)
  num_todo = len(dataset) if num_todo is None else num_todo
  perm = torch.randperm(len(dataset))
  todo_indices = perm[:num_todo]
  pbar = tqdm(todo_indices)

  all_ws, all_w_exps = [], []
  pbar = tqdm(range(num_todo))
  for i, todo_ind in enumerate(pbar):
    desc_str = f"squad {configs.desc_str()}"
    pbar.set_description(desc_str)
    datai = dataset[todo_ind]
    input_ids = datai[0].to(device)
    attn_mask = datai[1].to(device)
    start_pos = datai[3].to(device)
    end_pos = datai[4].to(device)
    inputs_embeds = embed_fn(input_ids)

    w = torch.zeros_like(input_ids)
    for j in range(torch.min(start_pos, end_pos), torch.max(start_pos, end_pos)):
      w[j] = 1

    ww_aexp = aexplainer.get_explanation(inputs_embeds.unsqueeze(0), start_pos.view(1,1)).squeeze().to(device)
    ww_bexp = bexplainer.get_explanation(inputs_embeds.unsqueeze(0), end_pos.view(1,1)).squeeze().to(device)

    if callable(post_process_fun):
      w_exp = post_process_fun(ww_aexp, ww_bexp)
    else:
      w_aexp = ww_aexp.max(dim=1).values.clamp(0,1) * attn_mask
      w_bexp = ww_bexp.max(dim=1).values.clamp(0,1) * attn_mask

      exp_start = w_aexp.argmax()
      exp_end = w_bexp.argmax()
      w_exp = torch.zeros_like(w)
      for j in range(torch.min(exp_start, exp_end), torch.max(exp_start, exp_end)):
        w_exp[j] = 1

    all_ws.append(w.cpu())
    all_w_exps.append(w_exp.cpu())
    if do_save and (i % save_every_k == 0 or len(pbar) - i < 2):
      model_class = model.__class__
      state_dict = model.state_dict()
      stuff = {
          "dataset" : dataset,
          "model_class" : model_class,
          "model_state_dict" : None if save_small else state_dict,
          "method" : configs.desc_str(),
          "num_total" : len(all_ws),
          "w_exps" : all_w_exps,
          "ws" : all_ws,
          "todo_indices" : todo_indices,
          "custom_desc" : custom_desc,
          "misc_data" : misc_data,
      }
      torch.save(stuff, saveto)

  return stuff


DEFAULT_MODELS_DIR = "saved_models"
DEFAULT_SAVETO_DIR = "saved_explanations"

def generate_explanations_sample_mvtec(
        seeds,
        methods_todo = ["grad", "intg", "lime", "shap"],
        num_todo = 100,
        state_dict_file = os.path.join(DEFAULT_MODELS_DIR, "sample_mvtec_epoch20.pt"),
        saveto_dir = DEFAULT_SAVETO_DIR):
  model = MyFastResA()
  if os.path.isfile(state_dict_file):
    model.load_state_dict(torch.load(state_dict_file))
    print(f"Loaded state dict from {state_dict_file}")

  bundle = get_data_bundle("mvtec")
  dataset = bundle["test_dataset"]
  grad_configs = GradConfigs()
  intg_configs = IntGradConfigs()
  lime_configs = LimeConfigs(x_train=torch.rand(500,3,256,256))
  shap_configs = ShapConfigs()

  for i, seed in enumerate(seeds):
    print(f"mvtec {i+1}/{len(seeds)}: using seed {seed}")
    if "grad" in methods_todo:
      grad_saveto = os.path.join(saveto_dir, f"mvtec_grad_seed{seed}.pt")
      get_mvtec_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed)

    if "intg" in methods_todo:
      intg_saveto = os.path.join(saveto_dir, f"mvtec_intg_seed{seed}.pt")
      get_mvtec_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed)

    if "lime" in methods_todo:
      lime_saveto = os.path.join(saveto_dir, f"mvtec_lime_seed{seed}.pt")
      get_mvtec_explanations(model, dataset, lime_configs, saveto=lime_saveto, num_todo=num_todo, seed=seed)

    if "shap" in methods_todo:
      shap_saveto = os.path.join(saveto_dir, f"mvtec_shap_seed{seed}.pt")
      get_mvtec_explanations(model, dataset, shap_configs, saveto=shap_saveto, num_todo=num_todo, seed=seed)


def generate_explanations_sample_timeseries(
        dataset_name,
        seeds,
        methods_todo = ["grad", "intg", "lime", "shap"],
        num_todo = 100,
        models_dir = DEFAULT_MODELS_DIR,
        saveto_dir = DEFAULT_SAVETO_DIR):
  dataset_name = dataset_name.lower()
  if dataset_name == "swat":
    model = MyLSTM(in_shape=(51,), return_mode="last")
    state_dict_file = os.path.join(models_dir, "sample_swat_epoch100.pt")
    lime_configs = LimeConfigs(x_train=torch.randn(50,100,51))
  elif dataset_name == "hai":
    model = MyLSTM(in_shape=(86,), return_mode="last")
    state_dict_file = os.path.join(models_dir, "sample_hai_epoch100.pt")
    lime_configs = LimeConfigs(x_train=torch.randn(50,100,86))
  elif dataset_name == "wadi":
    model = MyLSTM(in_shape=(127,), return_mode="last")
    state_dict_file = os.path.join(models_dir, "sample_wadi_epoch100.pt")
    lime_configs = LimeConfigs(x_train=torch.randn(50,100,127))
  else:
    raise NotImplementedError()

  if os.path.isfile(state_dict_file):
    model.load_state_dict(torch.load(state_dict_file))
    print(f"Loaded state dict from {state_dict_file}")
  
  bundle = get_data_bundle(dataset_name)
  dataset = bundle["train_dataset"]
  grad_configs = GradConfigs(train_mode=True)
  intg_configs = IntGradConfigs(train_mode=True)
  shap_configs = ShapConfigs()

  for i, seed in enumerate(seeds):
    print(f"{dataset_name} {i+1}/{len(seeds)}: using seed {seed}")
    if "grad" in methods_todo:
      grad_saveto = os.path.join(saveto_dir, f"{dataset_name}_grad_seed{seed}.pt")
      get_timeseries_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed)

    if "intg" in methods_todo:
      intg_saveto = os.path.join(saveto_dir, f"{dataset_name}_intg_seed{seed}.pt")
      get_timeseries_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed)

    if "lime" in methods_todo:
      lime_saveto = os.path.join(saveto_dir, f"{dataset_name}_lime_seed{seed}.pt")
      get_timeseries_explanations(model, dataset, lime_configs, saveto=lime_saveto, num_todo=num_todo, seed=seed)

    if "shap" in methods_todo:
      shap_saveto = os.path.join(saveto_dir, f"{dataset_name}_shap_seed{seed}.pt")
      get_timeseries_explanations(model, dataset, shap_configs, saveto=shap_saveto, num_todo=num_todo, seed=seed)


def generate_explanations_sample_squad(
        seeds,
        methods_todo = ["grad", "intg", "lime", "shap"],
        num_todo = 100,
        state_dict_file = os.path.join(DEFAULT_MODELS_DIR, "sample_squad_epoch5.pt"),
        saveto_dir = DEFAULT_SAVETO_DIR):

  bundle = get_data_bundle("squad", tokenizer_or_name="roberta-base")
  dataset = bundle["train_dataset"]
  tokenizer = bundle["tokenizer"]

  model = MySquad("roberta-base", tokenizer, input_mode="inputs_embeds")
  if os.path.isfile(state_dict_file):
    model.load_state_dict(torch.load(state_dict_file))
    print(f"Loaded state dict from {state_dict_file}")

  grad_configs = GradConfigs()
  intg_configs = IntGradConfigs()
  lime_configs = LimeConfigs(x_train=torch.randn(500,384))
  shap_configs = ShapConfigs()

  for i, seed in enumerate(seeds):
    print(f"squad {i+1}/{len(seeds)}: using seed {seed}")
    if "grad" in methods_todo:
      grad_saveto = os.path.join(saveto_dir, f"squad_grad_seed{seed}.pt")
      get_squad_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed)

    if "intg" in methods_todo:
      intg_saveto = os.path.join(saveto_dir, f"squad_intg_seed{seed}.pt")
      get_squad_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed)

    if "lime" in methods_todo:
      lime_saveto = os.path.join(saveto_dir, f"squad_lime_seed{seed}.pt")
      get_squad_explanations(model, dataset, lime_configs, saveto=lime_saveto, num_todo=num_todo, seed=seed)

    if "shap" in methods_todo:
      shap_saveto = os.path.join(saveto_dir, f"squad_shap_seed{seed}.pt")
      get_squad_explanations(model, dataset, shap_configs, saveto=shap_saveto, num_todo=num_todo, seed=seed)


