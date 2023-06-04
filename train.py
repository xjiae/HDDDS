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
               verbose = True):
    self.lr = lr
    self.num_epochs = num_epochs
    self.device_ids = device_ids
    self.verbose = verbose
    self.models_saveto_dir = models_saveto_dir


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
      loss = loss_fn((y_pred+1)/2, (y+1)/2) # Make -1/+1 value into 0/1 for BCE Loss
      if phase == "train":
        loss.backward()
        optimizer.step()
    num_processed += x.size(0)
    num_corrects += torch.sum(y == torch.sign(y_pred))
    running_loss += loss.item()
    avg_loss, avg_acc = (running_loss / num_processed), (num_corrects / num_processed)
    desc_str = f"[train]" if phase == "train" else "[valid]"
    desc_str += f" processed {num_processed}, loss {avg_loss:.4f}, acc {avg_acc:.4f}"
    pbar.set_description(desc_str)
  return model, running_loss / num_processed, num_corrects / num_processed # model, avg loss, avg acc


def train_mvtec(model, categories, configs, saveto_filename_prefix=None):
  optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
  train_dataset, valid_dataset, train_loader, valid_loader = get_mvtec_dataloaders(categories)
  categories_str = "_".join(categories)

  best_valid_acc = 0.0
  best_model_weights = copy.deepcopy(model.state_dict())

  print(f"Training MVTec with cats: {categories}")
  print(f"Will save to {configs.models_saveto_dir}")
  print(f"PID {os.getpid()}, num epochs {configs.num_epochs}, lr {configs.lr}")
  
  for epoch in range(1, configs.num_epochs+1):
    print(f"# epoch {epoch}/{configs.num_epochs}")
    train_stats = run_once_mvtec(model, train_loader, optimizer, "train", configs)
    valid_stats = run_once_mvtec(model, valid_loader, optimizer, "valid", configs)
    _, train_loss, train_acc = train_stats
    _, valid_loss, valid_acc = valid_stats
    desc_str = f"train (loss {train_loss:.4f}, acc {train_acc:.4f}), "
    desc_str += f"valid (loss {valid_loss:4f}, acc {valid_acc:.4f})"
    print(desc_str)

    saveto = f"mvtec_{categories_str}_epoch{epoch}.pt"
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


def get_tabular_dataloader(dataset,
                           train_batch_size = 32,
                           valid_batch_size = 32,
                           train_frac = 0.7,
                           mix_good_and_anom = True,
                           seed = None):
  total = len(dataset)
  num_train = int(total * train_frac)
  train_indices, valid_indices = list(range(total))[:num_train], list(range(total))[num_train:] 
  train_dataset = torch.utils.data.Subset(dataset, train_indices)
  valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)
  return train_dataset, valid_dataset, train_loader, valid_loader

def run_once_tabular(model, dataloader, optimizer, phase, configs):
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
    running_loss += loss.item()
    avg_loss = (running_loss / num_processed)
    desc_str = f"[train]" if phase == "train" else "[valid]"
    desc_str += f" processed {num_processed}, loss {avg_loss:.5f}"
    pbar.set_description(desc_str)
  return model, running_loss / num_processed # model, avg loss


def train_tabular(model, ds_name, dataset, configs, saveto_filename_prefix=None):
  # optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
  optimizer = torch.optim.AdamW(model.parameters())
  train_dataset, valid_dataset, train_loader, valid_loader = get_tabular_dataloader(dataset)
  
  best_valid_loss = sys.float_info.max
  best_model_weights = copy.deepcopy(model.state_dict())

  print(f"Training {ds_name}:")
  print(f"Will save to {configs.models_saveto_dir}")
  print(f"PID {os.getpid()}, num epochs {configs.num_epochs}, lr {configs.lr}")
  
  for epoch in range(1, configs.num_epochs+1):
    print(f"# epoch {epoch}/{configs.num_epochs}")
    train_stats = run_once_tabular(model, train_loader, optimizer, "train", configs)
    valid_stats = run_once_tabular(model, valid_loader, optimizer, "valid", configs)
    _, train_loss = train_stats
    _, valid_loss = valid_stats
    desc_str = f"train (loss {train_loss:.4f}) "
    desc_str += f"valid (loss {valid_loss:4f})"
    print(desc_str)

    saveto = f"{ds_name}_epoch{epoch}.pt"
    saveto = saveto if saveto_filename_prefix is None else saveto_filename_prefix + "_" + saveto
    saveto = os.path.join(configs.models_saveto_dir, saveto)
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      best_model_weights = copy.deepcopy(model.state_dict())
      torch.save(best_model_weights, saveto)
      print(f"Saved to {saveto}")

  # Save the best model so far at the end of training
  torch.save(best_model_weights, saveto)
  model.load_state_dict(best_model_weights)
  torch.cuda.empty_cache()
  return model.cpu()