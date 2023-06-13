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


def run_mvtec(seeds=[0,1], num_todo=5, model_filename="sample_mvtec_epoch3.pt"):
  model = MyFastResA()
  state_dict = torch.load(os.path.join(MODELS_DIR, model_filename))
  model.load_state_dict(state_dict)
  mvtec_bundle = get_data_bundle("mvtec")
  dataset = mvtec_bundle["test_dataset"]

  grad_configs = GradConfigs()
  intg_configs = IntGradConfigs()
  lime_configs = LimeConfigs(x_train=torch.rand(100,3,256,256))
  shap_configs = ShapConfigs()

  for i, seed in enumerate(seeds):
    print(f"mvtec: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"mvtec_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"mvtec_intg_seed{seed}.pt")
    lime_saveto = os.path.join(SAVETO_DIR, f"mvtec_lime_seed{seed}.pt")
    shap_saveto = os.path.join(SAVETO_DIR, f"mvtec_shap_seed{seed}.pt")
    get_mvtec_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed)
    get_mvtec_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed)
    get_mvtec_explanations(model, dataset, lime_configs, saveto=lime_saveto, num_todo=num_todo, seed=seed)
    get_mvtec_explanations(model, dataset, shap_configs, saveto=shap_saveto, num_todo=num_todo, seed=seed)


def run_hai(seeds=[0,1], num_todo=5, model_filename="sample_hai_epoch100.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_filename))
  model = MyLSTM(in_shape=(86,), return_mode="last")
  model.load_state_dict(state_dict)

  hai_bundle = get_data_bundle("hai", window_size=100)
  dataset = hai_bundle["test_dataset"]

  grad_configs = GradConfigs(train_mode=True)
  intg_configs = IntGradConfigs(train_mode=True)
  lime_configs = LimeConfigs(x_train=torch.rand(100,86))
  shap_configs = ShapConfigs()

  for i, seed in enumerate(seeds):
    print(f"hai: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"hai_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"hai_intg_seed{seed}.pt")
    lime_saveto = os.path.join(SAVETO_DIR, f"hai_lime_seed{seed}.pt")
    shap_saveto = os.path.join(SAVETO_DIR, f"hai_shap_seed{seed}.pt")
    get_timeseries_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_timeseries_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_timeseries_explanations(model, dataset, lime_configs, saveto=lime_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_timeseries_explanations(model, dataset, shap_configs, saveto=shap_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_wadi(seeds=[0,1], num_todo=100, model_filename="lr_wadi_epoch5.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_filename))
  model = LogisticRegression(in_shape=(127,), return_mode="last")
  model.load_state_dict(state_dict)
  dataset = WADIDataset(contents="all")
  dataset = get_timeseries_balanced_subset(dataset, seed=seeds[0])
  for i, seed in enumerate(seeds):
    print(f"wadi lr: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"wadi_lr_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"wadi_lr_intg_seed{seed}.pt")
    get_timeseries_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_timeseries_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_swat(seeds=[0,1], num_todo=100, model_filename="lr_swat_epoch5.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_filename))
  model = LogisticRegression(in_shape=(51,), return_mode="last")
  model.load_state_dict(state_dict)
  dataset = SWaTDataset(contents="all")
  dataset = get_timeseries_balanced_subset(dataset, seed=seeds[0])
  for i, seed in enumerate(seeds):
    print(f"swat lr: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"swat_lr_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"swat_lr_intg_seed{seed}.pt")
    get_timeseries_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_timeseries_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


def run_squad(seeds=[0,1], num_todo=100, model_filename="squad_epoch2.pt"):
  tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
  model = MySquadModel("roberta-base", tokenizer, input_mode="inputs_embeds")
  state_dict = torch.load(os.path.join(MODELS_DIR, model_filename))
  model.load_state_dict(state_dict)
  dataset = SquadDataset("roberta-base", is_train=False)
  for i, seed in enumerate(seeds):
    print(f"squad: using seed number {i+1}/{len(seeds)}")
    grad_saveto = os.path.join(SAVETO_DIR, f"squad_roberta_grad_seed{seed}.pt")
    intg_saveto = os.path.join(SAVETO_DIR, f"squad_roberta_intg_seed{seed}.pt")
    get_squad_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_squad_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)


if __name__ == "__main__":
  seeds = range(20)
  # run_wadi_sliding()
