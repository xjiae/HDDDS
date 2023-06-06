import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from models import *
from train import *
from dataset import *
from hddds import *
from get_explanations import *

simple_net = SimpleNet(in_shape=(3,24,24), out_shape=(10,))
my_ff = MyFastResA()

x = torch.rand(7,3,24,24)
xs = torch.rand(7,20,3,24,24)
xxs = torch.rand(7,3,256,256)

simple_net = SimpleNet(in_shape=(3,256,256), out_shape=(1,), softmax_output=True)
ffres = MyFastResA(return_mode="scalar_score")
hai_lstm = SimpleLSTM(in_shape=(86,), out_shape=(1,), return_mode="last", softmax_output=True)
swat_lstm = SimpleLSTM(in_shape=(51,), out_shape=(1,), return_mode="last", softmax_output=True)
wadi_lstm = SimpleLSTM(in_shape=(127,), out_shape=(1,), return_mode="last", softmax_output=True)

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
mysquad = MySquadModel("roberta-base", tokenizer, input_mode="dict")
embed_fn = mysquad.model.get_input_embeddings()

mvtec_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_MVTEC_LOADER_KWARGS)
hai_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_HAI_LOADER_KWARGS)
wadi_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_WADI_LOADER_KWARGS)
swat_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_SWAT_LOADER_KWARGS)
hai_sliding_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_HAI_SLIDING_LOADER_KWARGS)
wadi_sliding_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_WADI_SLIDING_LOADER_KWARGS)
swat_sliding_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_SWAT_SLIDING_LOADER_KWARGS)
squad_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_SQUAD_LOADER_KWARGS)

###

squad_stuff = get_squad_dataloaders()
squad_trains = squad_stuff["train_dataset"]
squad_valids = squad_stuff["valid_dataset"]

roberta = AutoModelForQuestionAnswering.from_pretrained("roberta-base")

context = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'

question = "'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'"

