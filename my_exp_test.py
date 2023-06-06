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
mysquad = MySquadModel("roberta-base", tokenizer, input_mode="input_ids")


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


context = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'

question = "'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'"

inputs = tokenizer(question, context)
input_ids = torch.tensor(inputs["input_ids"]).view(1,-1)


