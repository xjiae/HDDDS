import os
import copy
import torch
import torch.nn as nn
import sys
from tqdm import tqdm

from transformers import AutoTokenizer

from datasets import *
from models import *

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


# Run a single epoch for mvtec
def run_once_mvtec(model, dataloader, optimizer, phase, configs, device="cuda"):
  assert isinstance(configs, TrainConfigs) and phase in ["train", "test"]
  model = nn.DataParallel(model, device_ids=configs.device_ids)
  _ = model.train().to(device) if phase == "train" else model.eval().to(device)
  loss_fn = nn.CrossEntropyLoss(reduction="sum")
  num_processed, num_corrects = 0, 0
  running_loss, avg_acc = 0.0, 0.0
  pbar = tqdm(dataloader)
  for it, (x, y, w) in enumerate(pbar):
    _ = optimizer.zero_grad() if phase == "train" else None
    x, y, w = x.to(device), y.to(device), w.to(device)
    with torch.set_grad_enabled(phase == "train"):
      logits = model(x)
      loss = loss_fn(logits, y)
      if phase == "train":
        loss.backward()
        optimizer.step()
    num_processed += x.size(0)
    num_corrects += torch.sum(logits.argmax(dim=1) == y)
    running_loss += loss.item()
    avg_loss, avg_acc = (running_loss / num_processed), (num_corrects / num_processed)
    desc_str = f"[train]" if phase == "train" else "[test] "
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
  loss_fn = nn.CrossEntropyLoss(reduction="sum")
  num_processed, num_corrects = 0, 0
  running_loss, avg_acc = 0.0, 0.0
  pbar = tqdm(dataloader)
  for it, (x, y, w) in enumerate(pbar):
    _ = optimizer.zero_grad() if phase == "train" else None
    x, y, w = x.to(device), y.to(device), w.to(device)
    with torch.set_grad_enabled(phase == "train"):
      logits = model(x)
      loss = loss_fn(logits, y)
      if phase == "train":
        loss.backward()
        optimizer.step()
    num_processed += x.size(0)
    num_corrects += torch.sum(logits.argmax(dim=1) == y)
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

  # Only run squad in training mode; bail if test because it's too annoying to compute loss :)
  if phase == "test":
    return { "model" : model,
             "avg_loss" : 1.0,
             "avg_acc" : 0.0
           }

  model.train().to(device)
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

    with torch.set_grad_enabled(phase == "train"):
      outputs = model(**inputs)
      loss = outputs["loss"]  # This should be supplied by the model
      loss = loss.sum()
      loss.backward(retain_graph=True)
      optimizer.step()

    num_processed += batch[0].size(0)
    running_loss += loss.item()
    avg_loss = running_loss / num_processed
    desc_str = f"[train] processed {num_processed}, loss {avg_loss:.4f}"
    pbar.set_description(desc_str)

  return { "model" : model,
           "avg_loss" : running_loss / num_processed,
           "avg_acc" : 0.0 }

# Big train function
def train(model, dataset_name, configs,
          run_once_func=None,
          device = "cuda",
          save_when = "best_acc",
          saveto_filename_prefix = None):
  assert save_when in ["best_acc", "best_loss"]
  dataset_name = dataset_name.lower()

  # Get the data
  data_bundle = get_data_bundle(dataset_name, **configs.loaders_kwargs)
  train_loader = data_bundle["train_dataloader"]
  test_loader = data_bundle["test_dataloader"]

  if dataset_name == "mvtec":
    run_once_func = run_once_mvtec
  elif dataset_name in ["hai", "swat", "wadi"]:
    run_once_func = run_once_timeseries
  elif dataset_name == "squad":
    run_once_func = run_once_squad
  else:
    raise NotImplementedError()

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


DEFAULT_CONFIGS = TrainConfigs()
DEFAULT_MVTEC_LOADER_KWARGS = { "categories" : ["all"] }
DEFAULT_HAI_LOADER_KWARGS = { "window_size" : 100, "label_choice" : "last" }
DEFAULT_SWAT_LOADER_KWARGS = { "window_size" : 100, "label_choice" : "last" }
DEFAULT_WADI_LOADER_KWARGS = { "window_size" : 100, "label_choice" : "last" }
DEFAULT_SQUAD_LOADER_KWARGS = {}

# Call this method to auto-populate some sample models
def train_sample_model(dataset_name):
  dataset_name = dataset_name.lower()
  configs = DEFAULT_CONFIGS
  save_when = "best_acc"

  if dataset_name == "mvtec":
    model = MyFastResA()
    configs.num_epochs = 20
    configs.loaders_kwargs = DEFAULT_MVTEC_LOADER_KWARGS
  elif dataset_name == "hai":
    model = MyLSTM(in_shape=(86,), return_mode="last")
    configs.num_epochs = 100
    configs.loaders_kwargs = DEFAULT_HAI_LOADER_KWARGS
  elif dataset_name == "swat":
    model = MyLSTM(in_shape=(51,), return_mode="last")
    configs.num_epochs = 100
    configs.loaders_kwargs = DEFAULT_SWAT_LOADER_KWARGS
  elif dataset_name == "wadi":
    model = MyLSTM(in_shape=(127,), return_mode="last")
    configs.num_epochs = 100
    configs.loaders_kwargs = DEFAULT_WADI_LOADER_KWARGS
  elif dataset_name == "squad":
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
    model = MySquad("roberta-base", tokenizer)
    configs.num_epochs = 10
    configs.loaders_kwargs = DEFAULT_SQUAD_LOADER_KWARGS
    save_when = "best_loss"
  else:
    raise NotImplementedError()

  train(model, dataset_name,
        configs = configs,
        save_when = save_when,
        saveto_filename_prefix = "sample")
    
  return model


