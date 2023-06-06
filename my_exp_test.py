import torch
import torch.utils.data as tud
from transformers import AutoTokenizer

from models import *
from train import *
from get_explanations import *
from utils import *
mvtec_stuff = get_mvtec_dataloaders(["all"], mix_good_and_anom=True)
mvtec_dataset = mvtec_stuff["train_dataset"]

squad_stuff = get_squad_dataloaders()
squad_dataset = squad_stuff["train_dataset"]

simple_net = SimpleNet(in_shape=(3,256,256), out_shape=(2,))
ffres = MyFastResA(return_mode="two_class")

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
mysquad = MySquadModel("roberta-base", tokenizer, input_mode="input_ids")


# model = simple_net
model = ffres


grad_configs = GradConfigs()
intg_configs = IntGradConfigs()
lime_configs = LimeConfigs(x_train=torch.rand(100,3,256,256))
shap_configs = ShapConfigs()


def evaluate(ret, ds_name, model_name, attr_name):
  w_true = torch.cat(ret['ws']).numpy().flatten()
  w_pred = torch.cat(ret['w_exps']).numpy().flatten()
  acc, f1, fpr, fnr = summary(w_true, w_pred, score=True)
  saveto = open(f"results/results_{attr_name}.txt", "a")
  saveto.write(f"{ds_name},{model_name},{fpr:.4f},{fnr:.4f},{acc:.4f},{f1:.4f}\n")
  saveto.close()
  return acc, f1, fpr, fnr 

def run_mvtec(num_todo=250, seed=1234):
  grad_saveto = "saved_explanations/mvtec_grad.pt"
  intg_saveto = "saved_explanations/mvtec_intg.pt"
  # lime_saveto = "saved_explanations/mvtec_lime.pt"
  # shap_saveto = "saved_explanations/mvtec_shap.pt"

  grad_stuff = get_mvtec_explanation(model, mvtec_dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed)
  intg_stuff = get_mvtec_explanation(model, mvtec_dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo, seed=seed)
  # lime_stuff = get_mvtec_explanation(model, mvtec_dataset, lime_configs, saveto=lime_saveto, num_todo=num_todo, seed=seed)
  # shap_stuff = get_mvtec_explanation(model, mvtec_dataset, shap_configs, saveto=shap_saveto, num_todo=num_todo, seed=seed)
  # return grad_stuff, intg_stuff, lime_stuff, shap_stuff
  
  
  return grad_stuff, intg_stuff

def run_tabular(ds_name, model_name, dataset, fea_num, num_todo=1000, seed=1234):
  grad_saveto = f"saved_explanations/{ds_name}_{model_name}_grad.pt"
  if model_name == "lr":
    model = LogisticRegression(in_shape=(fea_num,), out_shape=(1,), return_mode="two_class")
  else: 
    return
  model_path = open(f"saved_models/{model_name}_{ds_name}_epoch5.pt", "rb") 
  load = torch.load(model_path)
  model.load_state_dict(load)
  grad_stuff = get_tabular_explanation(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed)
  evaluate(grad_stuff, ds_name, model_name, "grad")
 
def run_sliding_tabular(ds_name, model_name, dataset, fea_num, num_todo=1000, seed=1234):
  grad_saveto = f"saved_explanations/{ds_name}_{model_name}_grad.pt"
  if model_name == "lstm":
    model = SimpleLSTM(in_shape=(fea_num,), out_shape=(1,), return_mode="two_class")
  else: 
    return
  model_path = open(f"saved_models/{model_name}_{ds_name}-sliding_epoch5.pt", "rb") 
  load = torch.load(model_path)
  model.load_state_dict(load)
  grad_stuff = get_tabular_sliding_explanation(model, dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo, seed=seed)
  evaluate(grad_stuff, ds_name, model_name, "grad")
  
  
context = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'

question = "'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'"

inputs = tokenizer(question, context)
input_ids = torch.tensor(inputs["input_ids"]).view(1,-1)


if __name__=="__main__":
      seeds = list(range(1234, 1334))
      ds = ["hai", "swat", "wadi"]
      for d in ds:
        stuff = None
        sstuff = None
        match d:
          case "hai":
            stuff = get_hai_dataloaders() 
            sstuff = get_hai_sliding_dataloaders(100) 
          case "swat":
            stuff = get_swat_dataloaders()
            sstuff = get_swat_sliding_dataloaders(100) 
          case "wadi":
            stuff = get_wadi_dataloaders()
            sstuff = get_wadi_sliding_dataloaders(100) 
        dataset = stuff["valid_dataset"]
        sdataset = sstuff["valid_dataset"]
        for seed in seeds:
          # run_mvtec(num_todo=250, seed=seed)
          run_tabular(d, "lr", dataset, num_todo=250, seed=seed)
          run_tabular(d, "lstm", sdataset, num_todo=250, seed=seed)
        
