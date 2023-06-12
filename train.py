import os
import copy
import torch
import torch.nn as nn
import sys
from tqdm import tqdm

from datasets import *

DEFAULT_MODELS_SAVETO_DIR = "saved_models"

class TrainConfigs:
  def __init__(self,
               lr = 1e-6,
               num_epochs = 2,
               device_ids = None, # Which GPUs are we using if data parallel
               models_saveto_dir = DEFAULT_MODELS_SAVETO_DIR,
               loaders_kwargs = {}):
    self.lr = lr
    self.num_epochs = num_epochs
    self.device_ids = device_ids
    self.models_saveto_dir = models_saveto_dir
    self.loaders_kwargs = loaders_kwargs

DEFAULT_CONFIGS = TrainConfigs()
DEFAULT_MVTEC_LOADER_KWARGS = { "categories" : ["all"] }
DEFAULT_HAI_LOADER_KWARGS = {}
DEFAULT_SWAT_LOADER_KWARGS = { "window_size" : 100 }
DEFAULT_WADI_LOADER_KWARGS = { "window_size" : 100 }
DEFAULT_SQUAD_LOADER_KWARGS = { "window_size" : 100 }

# Run a single epoch for mvtec
def run_once_mvtec(model, dataloader, optimizer, phase, configs, device="cuda"):
  assert isinstance(configs, TrainConfigs) and phase in ["train", "test"]
  model = nn.DataParallel(model, device_ids=configs.device_ids)
  _ = model.train().to(device) if phase == "train" else model.eval().to(device)
  loss_fn = nn.BCELoss(reduction="sum")
  num_processed, num_corrects = 0, 0
  running_loss, avg_acc = 0.0, 0.0
  pbar = tqdm(dataloader)
  for it, (x, y, w) in enumerate(pbar):
    _ = optimizer.zero_grad() if phase == "train" else None
    x, y, w = x.to(device), y.to(device), w.to(device)
    with torch.set_grad_enabled(phase == "train"):
      y_pred = model(x).view(-1)
      loss = loss_fn(y_pred.double(), y.double())
      if phase == "train":
        loss.backward()
        optimizer.step()
    num_processed += x.size(0)
    num_corrects += torch.sum(y == (y_pred > 0.5))
    running_loss += loss.item()
    avg_loss, avg_acc = (running_loss / num_processed), (num_corrects / num_processed)
    desc_str = f"[train]" if phase == "train" else "[test]"
    desc_str += f" processed {num_processed}, loss {avg_loss:.4f}, acc {avg_acc:.4f}"
    pbar.set_description(desc_str)
  return { "model" : model,
           "avg_loss" : running_loss / num_processed,
           "avg_acc" : num_corrects / num_processed }


# Sliding tbular stuff
def run_once_timeseries(model, dataloader, optimizer, phase, configs, device="cuda"):
  assert isinstance(configs, TrainConfigs) and phase in ["train", "test"]
  model = nn.DataParallel(model, device_ids=configs.device_ids)
  _ = model.train().to(device) if phase == "train" else model.eval().to(device)

  loss_fn = nn.BCELoss(reduction="sum")
  num_processed, num_corrects = 0, 0
  running_loss, avg_acc = 0.0, 0.0
  pbar = tqdm(dataloader)
  for it, (x, y, w, l) in enumerate(pbar):
    _ = optimizer.zero_grad() if phase == "train" else None
    x, y, w, l = x.to(device), y.to(device), w.to(device), l.to(device)
    with torch.set_grad_enabled(phase == "train"):
      y_pred = model(x).view(y.shape) # Assume already outputs in [0,1]
      loss = loss_fn(y_pred.double(), y.double())
      if phase == "train":
        loss.backward()
        optimizer.step()
    num_processed += x.size(0)
    num_corrects += torch.sum(y == (y_pred > 0.5))
    running_loss += loss.item()
    avg_loss, avg_acc = (running_loss / num_processed), (num_corrects / num_processed)
    desc_str = f"[train]" if phase == "train" else "[test] "
    desc_str += f" processed {num_processed}, loss {avg_loss:.5f}, acc {avg_acc:.4f}"
    pbar.set_description(desc_str)
  return { "model" : model,
           "avg_loss" : running_loss / num_processed,
           "avg_acc" : num_corrects / num_processed }


# squad training
def run_once_squad(model, dataloader, optimizer, phase, configs, device="cuda"):
  assert isinstance(configs, TrainConfigs) and phase in ["train", "test"]
  # model = nn.DataParallel(model, device_ids=configs.device_ids)
  _ = model.train().to(device) if phase == "train" else model.eval().to(device)

  num_processed, running_loss = 0, 0.0
  pbar = tqdm(dataloader)
  for it, batch in enumerate(pbar):
    _ = optimizer.zero_grad() if phase == "train" else None
    batch = tuple(b.to(device) for b in batch)
    inputs = {
      "input_ids": batch[0],
      "attention_mask": batch[1],
      "token_type_ids": batch[2],
      "start_positions": batch[3],  # Giving start and end position should have model yield loss
      "end_positions": batch[4],
    }


# Big train function
def train(model, dataset_name, configs=DEFAULT_CONFIGS,
          device = "cuda",
          save_when = "best_acc",
          saveto_filename_prefix = None):
  assert save_when in ["best_acc", "best_loss"]
  dataset_name = dataset_name.lower()

  if dataset_name == "mvtec":
    get_loaders_func = get_mvtec_dataloaders
    run_once_func = run_once_mvtec
    if len(configs.loaders_kwargs) == 0:
      configs.loaders_kwargs = DEFAULT_MVTEC_LOADER_KWARGS

  elif dataset_name == "hai":
    get_loaders_func = get_hai_dataloaders
    run_once_func = run_once_timeseries
    if len(configs.loaders_kwargs) == 0:
      configs.loaders_kwargs = DEFAULT_HAI_LOADER_KWARGS

  elif dataset_name == "swat":
    get_loaders_func = get_swat_dataloaders
    run_once_func = run_once_timeseries
    if len(configs.loaders_kwargs) == 0:
      configs.loaders_kwargs = DEFAULT_SWAT_LOADER_KWARGS

  elif dataset_name == "wadi":
    get_loaders_func = get_wadi_dataloaders
    run_once_func = run_once_timeseries
    if len(configs.loaders_kwargs) == 0:
      configs.loaders_kwargs = DEFAULT_WADI_LOADER_KWARGS

  elif dataset_name == "squad":
    get_loaders_func = get_squad_dataloaders
    run_once_func = run_once_squad
    if len(configs.loaders_kwargs) == 0:
      configs.loaders_kwargs = DEFAULT_SQUAD_LOADER_KWARGS

  else:
    raise NotImplementedError()

  loaders_dict = get_loaders_func(**configs.loaders_kwargs)
  train_loader = loaders_dict["train_dataloader"]
  test_loader = loaders_dict["test_dataloader"]
  optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

  best_test_loss, best_test_acc = 0.0, 0.0
  best_model_weights = copy.deepcopy(model.state_dict())

  print(f"Training with {dataset_name}")
  print(f"Will save to {configs.models_saveto_dir}")
  print(f"PID {os.getpid()}, num epochs {configs.num_epochs}, lr {configs.lr}")
  
  for epoch in range(1, configs.num_epochs+1):
    print(f"# epoch {epoch}/{configs.num_epochs}")
    train_stats = run_once_func(model, train_loader, optimizer, "train", configs, device=device)
    test_stats = run_once_func(model, test_loader, optimizer, "test", configs, device=device)
    train_loss, train_acc = train_stats["avg_loss"], train_stats["avg_acc"]
    test_loss, test_acc = test_stats["avg_loss"], test_stats["avg_acc"]
    desc_str = f"train (loss {train_loss:.4f}, acc {train_acc:.4f}), "
    desc_str += f"test (loss {test_loss:4f}, acc {test_acc:.4f})"
    print(desc_str)

    saveto = f"{dataset_name}_epoch{epoch}.pt"
    saveto = saveto if saveto_filename_prefix is None else saveto_filename_prefix + "_" + saveto
    saveto = os.path.join(configs.models_saveto_dir, saveto)
    if ((save_when == "best_acc" and test_acc >= best_test_acc)
        or (save_when == "best_loss" and test_loss <= best_test_loss)):
      best_test_acc = test_acc
      best_model_weights = copy.deepcopy(model.state_dict())
      torch.save(best_model_weights, saveto)
      print(f"Saved to {saveto}")

  # Save the best model so far at the end of training
  torch.save(best_model_weights, saveto)
  model.load_state_dict(best_model_weights)
  torch.cuda.empty_cache()
  return model.cpu()



