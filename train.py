import os
import copy
import torch
import torch.nn as nn
import sys
from tqdm import tqdm

from dataset import *

DEFAULT_MODELS_SAVETO_DIR = "saved_models"

class TrainConfigs:
  def __init__(self,
               lr = 1e-6,
               num_epochs = 5,
               device_ids = None, # Which GPUs are we using if data parallel
               models_saveto_dir = DEFAULT_MODELS_SAVETO_DIR,
               loaders_kwargs = {}):
    self.lr = lr
    self.num_epochs = num_epochs
    self.device_ids = device_ids
    self.models_saveto_dir = models_saveto_dir
    self.loaders_kwargs = loaders_kwargs

DEFAULT_MVTEC_LOADER_CONFIGS = { "categories" : ["all"] }
DEFAULT_HAI_LOADER_CONFIGS = {}
DEFAULT_SWAT_LOADER_CONFIGS = {}
DEFAULT_WADI_LOADER_CONFIGS = {}
DEFAULT_HAI_SLIDING_LOADER_CONFIGS = { "window_size" : 100 }
DEFAULT_SWAT_SLIDING_LOADER_CONFIGS = { "window_size" : 100}
DEFAULT_WADI_SLIDING_LOADER_CONFIGS = { "window_size" : 100 }
DEFAULT_CUAD_LOADER_CONFIGS = {}


# Run a single epoch for mvtec
def run_once_mvtec(model, dataloader, optimizer, phase, configs):
  assert isinstance(configs, TrainConfigs) and phase in ["train", "val", "valid"]
  model = nn.DataParallel(model, device_ids=configs.device_ids)
  _ = model.train().cuda() if phase == "train" else model.eval().cuda()
  loss_fn = nn.BCELoss(reduction="sum")
  num_processed, num_corrects = 0, 0
  running_loss, avg_acc = 0.0, 0.0
  pbar = tqdm(dataloader)
  for it, (x, y, w) in enumerate(pbar):
    _ = optimizer.zero_grad() if phase == "train" else None
    x, y, w = x.cuda(), y.cuda(), w.cuda()
    with torch.set_grad_enabled(phase == "train"):
      y_pred = model(x).view(-1)
      loss = loss_fn(y_pred, y)
      if phase == "train":
        loss.backward()
        optimizer.step()
    num_processed += x.size(0)
    num_corrects += torch.sum(y == (y_pred > 0.5))
    running_loss += loss.item()
    avg_loss, avg_acc = (running_loss / num_processed), (num_corrects / num_processed)
    desc_str = f"[train]" if phase == "train" else "[valid]"
    desc_str += f" processed {num_processed}, loss {avg_loss:.4f}, acc {avg_acc:.4f}"
    pbar.set_description(desc_str)
  return {
      "model" : model,
      "avg_loss" : running_loss / num_processed,
      "avg_acc" : running_corrects / num_processed
    }


# Sliding tbular stuff
def run_once_sliding_tabular(model, dataloader, optimizer, phase, configs):
  assert isinstance(configs, TrainConfigs) and phase in ["train", "val", "valid"]
  model = nn.DataParallel(model, device_ids=configs.device_ids)
  _ = model.train().cuda() if phase == "train" else model.eval().cuda()

  loss_fn = nn.BCELoss(reduction="sum")
  num_processed, num_corrects = 0, 0
  running_loss, avg_acc = 0.0, 0.0
  pbar = tqdm(dataloader)
  for it, (x, y, w, l) in enumerate(pbar):
    _ = optimizer.zero_grad() if phase == "train" else None
    x, y, w, l = x.cuda(), y.cuda(), w.cuda(), l.cuda()
    with torch.set_grad_enabled(phase == "train"):
      y_pred = torch.sigmoid(model(x).view(y.shape))
      
      loss = loss_fn(y_pred.double(), y.double())
      # breakpoint()
      if phase == "train":
        loss.backward()
        optimizer.step()
    num_processed += x.size(0)
    num_corrects += torch.sum(y == (y_pred > 0.5)) 
    running_loss += loss.item()
    avg_loss = (running_loss / num_processed)
    desc_str = f"[train]" if phase == "train" else "[valid]"
    desc_str += f" processed {num_processed}, loss {avg_loss:.5f}"
    pbar.set_description(desc_str)
  return model, running_loss / num_processed # model, avg loss

  return {
      "model" : model,
      "avg_loss" : running_loss / num_processed,
      "avg_acc" : running_corrects / num_processed
    }

# Big train function
def train(model, dataset_name, configs, saveto_prefix=None):
  if dataset_name == "mvtec":
    get_loaders_func = get_mvtec_dataloaders
    run_once_func = run_once_mvtec

  elif dataset_name == "hai":
    get_loaders_func = get_hai_dataloaders
    run_once_func = run_once_sliding_tabular

  elif dataset_name == "hai-sliding":
    get_loaders_func = get_hai_sliding_dataloaders
    run_once_func = run_once_sliding_tabular

  elif dataset_name == "swat":
    get_loaders_func = get_swat_dataloaders
    run_once_func = run_once_sliding_tabular

  elif dataset_name == "swat-sliding":
    get_loaders_func = get_swat_sliding_dataloaders
    run_once_func = run_once_sliding_tabular

  elif dataset_name == "wadi":
    get_loaders_func = get_wadi_dataloaders
    run_once_func = run_once_sliding_tabular

  elif dataset_name == "wadi-sliding":
    get_loaders_func = get_wadi_sliding_dataloaders
    run_once_func = run_once_sliding_tabular

  else:
    raise NotImplementedError()

  loaders_dict = get_loaders_func(**configs.loaders_kwargs)
  train_loader = loaders_dict["train_dataloader"]
  valid_loader = loaders_dict["valid_dataloader"]
  optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

  best_valid_acc = 0.0
  best_model_weights = copy.deepcopy(model.state_dict())

  print(f"Training with {dataset_name}")
  print(f"Will save to {configs.models_saveto_dir}")
  print(f"PID {os.getpid()}, num epochs {configs.num_epochs}, lr {configs.lr}")
  
  for epoch in range(1, configs.num_epochs+1):
    print(f"# epoch {epoch}/{configs.num_epochs}")
    train_stats = run_once_func(model, train_loader, optimizer, "train", configs)
    valid_stats = run_once_func(model, valid_loader, optimizer, "valid", configs)
    _, train_loss, train_acc = train_stats
    _, valid_loss, valid_acc = valid_stats
    desc_str = f"train (loss {train_loss:.4f}, acc {train_acc:.4f}), "
    desc_str += f"valid (loss {valid_loss:4f}, acc {valid_acc:.4f})"
    print(desc_str)

    saveto = f"{dataset_name}_epoch{epoch}.pt"
    saveto = saveto if saveto_filename_prefix is None else saveto_filename_prefix + "_" + saveto
    saveto = os.path.join(configs.models_saveto_dir, saveto)
    if valid_acc > best_valid_acc:
      best_valid_acc = valid_acc
      best_model_weights = copy.deepcopy(model.state_dict())
      torch.save(best_model_weights, saveto)
      print(f"Saved to {saveto}")

  # Save the best model so far at the end of training
  torch.save(best_model_weights, saveto)
  model.load_state_dict(best_model_weights)
  torch.cuda.empty_cache()
  return model.cpu()



