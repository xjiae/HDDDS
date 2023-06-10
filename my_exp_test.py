import os
import torch
import torch.utils.data as tud
from transformers import AutoTokenizer

from models import *
from train import *
from get_explanations import *


MODELS_DIR = "saved_models"
SAVETO_DIR = "saved_explanations"
assert os.path.isdir(MODELS_DIR) and os.path.isdir(SAVETO_DIR)


grad_configs = GradConfigs()
intg_configs = IntGradConfigs()




def run_mvtec(seeds=[0,1], num_todo=100, model_file="mvtec_epoch2.pt"):
  model = MyFastResA(return_mode="two_class")
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model.load_state_dict(state_dict)
  mvtec_stuff = get_mvtec_dataloaders(["all"], mix_good_and_anom=True)
  dataset = mvtec_stuff["valid_dataset"]
  for i, seed in enumerate(seeds):
    print(f"mvtec: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"mvtec_ffres_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"mvtec_ffres_intg_seed{seed}.pt")
    get_mvtec_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_mvtec_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_squad(seeds=[0,1], num_todo=100, model_file="squad_epoch2.pt"):
  tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
  model = MySquadModel("roberta-base", tokenizer, input_mode="inputs_embeds")
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model.load_state_dict(state_dict)
  dataset = SquadDataset("roberta-base", is_train=False)
  for i, seed in enumerate(seeds):
    print(f"squad: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"squad_roberta_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"squad_roberta_intg_seed{seed}.pt")
    get_squad_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_squad_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_hai(seeds=[0,1], num_todo=100, model_file="lr_hai_epoch5.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model = LogisticRegression(in_shape=(86,), return_mode="two_class")
  model.load_state_dict(state_dict)
  dataset = HAIDataset(contents="all")
  dataset = get_tabular_balanced_subset(dataset, seed=seeds[0])
  for i, seed in enumerate(seeds):
    print(f"hai lr: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"hai_lr_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"hai_lr_intg_seed{seed}.pt")
    get_tabular_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_tabular_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_wadi(seeds=[0,1], num_todo=100, model_file="lr_wadi_epoch5.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model = LogisticRegression(in_shape=(127,), return_mode="two_class")
  model.load_state_dict(state_dict)
  dataset = WADIDataset(contents="all")
  dataset = get_tabular_balanced_subset(dataset, seed=seeds[0])
  for i, seed in enumerate(seeds):
    print(f"wadi lr: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"wadi_lr_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"wadi_lr_intg_seed{seed}.pt")
    get_tabular_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_tabular_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_swat(seeds=[0,1], num_todo=100, model_file="lr_swat_epoch5.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model = LogisticRegression(in_shape=(51,), return_mode="two_class")
  model.load_state_dict(state_dict)
  dataset = SWaTDataset(contents="all")
  dataset = get_tabular_balanced_subset(dataset, seed=seeds[0])
  for i, seed in enumerate(seeds):
    print(f"swat lr: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"swat_lr_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"swat_lr_intg_seed{seed}.pt")
    get_tabular_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_tabular_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_hai_sliding(seeds=[0,1], num_todo=100, model_file="lstm_hai-sliding_epoch5.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model = SimpleLSTM(in_shape=(86,), out_shape=(1,), return_mode="two_class")
  model.load_state_dict(state_dict)
  dataset = HAISlidingDataset(window_size=100, contents="all")
  dataset = get_tabular_balanced_subset(dataset, seed=seeds[0])
  grad_configs.train_mode = True
  intg_configs.train_mode = True
  for i, seed in enumerate(seeds):
    print(f"hai lstm: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"hai_lstm_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"hai_lstm_intg_seed{seed}.pt")
    get_tabular_sliding_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_tabular_sliding_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_wadi_sliding(seeds=[0,1], num_todo=100, model_file="lstm_wadi-sliding_epoch5.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model = SimpleLSTM(in_shape=(127,), out_shape=(1,), return_mode="two_class")
  model.load_state_dict(state_dict)
  dataset = WADISlidingDataset(window_size=100, contents="all")
  dataset = get_tabular_balanced_subset(dataset, seed=seeds[0])
  grad_configs.train_mode = True
  intg_configs.train_mode = True
  for i, seed in enumerate(seeds):
    print(f"wadi_lstm: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"wadi_lstm_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"wadi_lstm_intg_seed{seed}.pt")
    get_tabular_sliding_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_tabular_sliding_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_swat_sliding(seeds=[0,1], num_todo=100, model_file="lstm_swat-sliding_epoch5.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model = SimpleLSTM(in_shape=(51,), out_shape=(1,), return_mode="two_class")
  model.load_state_dict(state_dict)
  dataset = SWaTSlidingDataset(window_size=100, contents="all")
  dataset = get_tabular_balanced_subset(dataset, seed=seeds[0])
  grad_configs.train_mode = True
  intg_configs.train_mode = True
  for i, seed in enumerate(seeds):
    print(f"swat_lstm: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"swat_lstm_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"swat_lstm_intg_seed{seed}.pt")
    get_tabular_sliding_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_tabular_sliding_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


if __name__ == "__main__":
  seeds = range(20)
  # run_wadi_sliding()
