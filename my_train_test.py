import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from models import *
from train import *
from datasets import *
from hddds import *
from get_explanations import *

simple_net = MySimpleNet(in_shape=(3,256,256), out_shape=(1,), softmax_output=True)
ffres = MyFastResA(return_mode="scalar_score")
hai_lstm = MyLSTM(in_shape=(86,))
swat_lstm = MyLSTM(in_shape=(51,))
wadi_lstm = MyLSTM(in_shape=(127,))

hai_lr = MyLogisticRegression(in_shape=(86,))
swat_lr = MyLogisticRegression(in_shape=(51,))
wadi_lr = MyLogisticRegression(in_shape=(127,))

squad_bundle = get_data_bundle("squad", tokenizer_or_name="roberta-base")
tokenizer = squad_bundle["tokenizer"]
mysquad = MySquad("roberta-base", tokenizer)

mvtec_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_MVTEC_LOADER_KWARGS)
hai_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_HAI_LOADER_KWARGS)
wadi_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_WADI_LOADER_KWARGS)
swat_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_SWAT_LOADER_KWARGS)
hai_sliding_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_HAI_SLIDING_LOADER_KWARGS)
wadi_sliding_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_WADI_SLIDING_LOADER_KWARGS)
swat_sliding_configs = TrainConfigs(num_epochs=5, loaders_kwargs=DEFAULT_SWAT_SLIDING_LOADER_KWARGS)
squad_configs = TrainConfigs(num_epochs=2, loaders_kwargs=DEFAULT_SQUAD_LOADER_KWARGS)

def run_to_train():
  train(swat_lstm, 'swat-sliding', swat_sliding_configs, saveto_filename_prefix="lstm")
  train(wadi_lstm, 'wadi-sliding', wadi_sliding_configs, saveto_filename_prefix="lstm")
  train(hai_lstm, 'hai-sliding', hai_sliding_configs, saveto_filename_prefix="lstm")



