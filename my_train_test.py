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
hai_lr = LogisticRegression(in_shape=(86,), out_shape=(1,))
swat_lr = LogisticRegression(in_shape=(51,), out_shape=(1,))
wadi_lr = LogisticRegression(in_shape=(127,), out_shape=(1,))

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
mysquad = MySquadModel("roberta-base", tokenizer, input_mode="dict")
embed_fn = mysquad.model.get_input_embeddings()

mvtec_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_MVTEC_LOADER_KWARGS)
hai_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_HAI_LOADER_KWARGS)
wadi_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_WADI_LOADER_KWARGS)
swat_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_SWAT_LOADER_KWARGS)
hai_sliding_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_HAI_SLIDING_LOADER_KWARGS)
wadi_sliding_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_WADI_SLIDING_LOADER_KWARGS)
swat_sliding_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_SWAT_SLIDING_LOADER_KWARGS)
squad_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_SQUAD_LOADER_KWARGS)

train(swat_lstm, 'swat-sliding', swat_sliding_configs, saveto_filename_prefix="lstm")
train(wadi_lstm, 'wadi-sliding', wadi_sliding_configs, saveto_filename_prefix="lstm")
train(hai_lstm, 'hai-sliding', hai_sliding_configs, saveto_filename_prefix="lstm")
