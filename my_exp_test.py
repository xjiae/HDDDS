import torch
import torch.utils.data as tud
from transformers import AutoTokenizer

from models import *
from train import *
from get_explanations import *

mvtec_stuff = get_mvtec_dataloaders(["all"], mix_good_and_anom=True)
mvtec_dataset = mvtec_stuff["train_dataset"]

squad_stuff = get_squad_dataloaders()
squad_dataset = squad_stuff["train_dataset"]

simple_net = SimpleNet(in_shape=(3,256,256), out_shape=(2,))
ffres = MyFastResA(return_mode="two_class")

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
mysquad = MySquadModel("roberta-base", input_mode="inputs_embeds")
embed_fn = mysquad.model.get_input_embeddings()


# model = simple_net
model = ffres

grad_configs = GradConfigs()
intg_configs = IntGradConfigs()
lime_configs = LimeConfigs(x_train=torch.rand(100,3,256,256))
shap_configs = ShapConfigs()


def run_mvtec(num_todo=250):
  grad_saveto = "saved_explanations/mvtec_grad.pt"
  intg_saveto = "saved_explanations/mvtec_intg.pt"
  lime_saveto = "saved_explanations/mvtec_lime.pt"
  shap_saveto = "saved_explanations/mvtec_shap.pt"

  grad_stuff = get_mvtec_explanation(model, mvtec_dataset, grad_configs, saveto=grad_saveto, num_todo=num_todo)
  intg_stuff = get_mvtec_explanation(model, mvtec_dataset, intg_configs, saveto=intg_saveto, num_todo=num_todo)
  lime_stuff = get_mvtec_explanation(model, mvtec_dataset, lime_configs, saveto=lime_saveto, num_todo=num_todo)
  shap_stuff = get_mvtec_explanation(model, mvtec_dataset, shap_configs, saveto=shap_saveto, num_todo=num_todo)
  return grad_stuff, intg_stuff, lime_stuff, shap_stuff


