import torch
import torch.utils.data as tud
from transformers import AutoTokenizer

from models import *
from train import *
from get_explanations import *
from utils import *


MODELS_DIR = "saved_models"

grad_configs = GradConfigs()
intg_configs = IntGradConfigs()

def evaluate(ret, seed):
  w_true = torch.cat(ret['ws']).numpy().flatten()
  w_pred = torch.cat(ret['w_exps']).numpy().flatten()
  acc, f1, fpr, fnr = summary(w_true, w_pred, score=True)
  saveto = open(f"results/results_seed={seed}.txt", "a")
  saveto.write(f"{model} & {exp} & {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}\n")
  saveto.close()
  return acc, f1, fpr, fnr 


def run_mvtec(seeds=[0,1], num_todo=250, model_file="mvtec_epoch2.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model = MyFastResA(return_mode="two_class")
  model.load_state_dict(state_dict)

  mvtec_stuff = get_mvtec_dataloaders(["all"], mix_good_and_anom=True)
  dataset = mvtec_stuff["train_dataset"]
  for i, seed in enumerate(seeds):
    print(f"mvtec: using seed number {i+1}/{len(seeds)}")
    grad_saveto = f"saved_explanations/mvtec_ffres_grad_seed{seed}.pt"
    intg_saveto = f"saved_explanations/mvtec_ffres_intg_seed{seed}.pt"
    get_mvtec_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_mvtec_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)

def run_hai_sliding(seeds=[0,1], num_todo=250, model_file="lr_hai_epoch5.pt"):
  model = 

def run_squad(seeds=[0,1], num_todo=250, model_file="squad_epochs2.pt"):
  state_dict = torch.load(os.path.join(MODELS_DIR, model_file))
  model = MySquadModel("roberta-base", tokenizer, input_mode="inputs_embeds")
  model.load_state_dict(state_dict)

  grad_saveto = "saved_explanations/squad_grad.pt"
  intg_saveto = "saved_explanations/squad_intg.pt"
  for seed in seeds:
    print(f"squad: using seed number {i+1}/{len(seeds)}")
    grad_saveto = f"saved_explanations/squad_roberta_grad_seed{seed}.pt"
    intg_saveto = f"saved_explanations/squad_roberta_intg_seed{seed}.pt"
    get_squad_explanations(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed, save_small=True)
    get_squad_explanations(model, dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed, save_small=True)



def run_sliding_tabular(num_to, seed=1234):
  lime_stuff = get_tabular_explanation



if __name__=="__main__":
  seeds = range(100)

