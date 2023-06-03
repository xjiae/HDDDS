import torch

from models import *
from train import *
from get_explanations import *
from hddds import *
import os

    
def explain(ds_name, exp):
    
    data = get_dataset(ds_name)
    
    config = None
    saveto = os.path.join("saved_explanations", f"{exp}_{ds_name}.pt")
    ret_mode = "last"
    device = "cuda"
    
    match exp:
        case "grad":
            # for lstm, need to be in train mode
            config = GradConfigs(train_mode=True)
            ret_mode = "mean"
        case "intg":
            # for lstm, need to be in train mode
            config = IntGradConfigs(train_mode=True)
            ret_mode = "mean"
        case "lime":
            config = LimeConfigs(x_train=torch.rand(100,*data[0][0].shape))
            ret_mode = "prob"
        case "shap":
            config = ShapConfigs()
            ret_mode = "mean"
            device = "cpu"
    model = SimpleLSTM(in_shape=(data.window_size-1, data.num_features), out_shape=(1,data.num_features), return_mode=ret_mode)
    mp = ds_name.replace("test", "train")
    model_path = open(f"saved_models/{mp}_epoch3.pt", "rb") 
    load = torch.load(model_path)
    model.load_state_dict(load)
    ret_dict = get_tabular_explanation(model, data, config, saveto=saveto, device=device, num_todo=1000)

    return ret_dict

if __name__== "__main__":

    data = ["hai_sliding_100_test","swat_sliding_100_test", "wadi_sliding_100_test"]
    exps = ["grad","intg", "lime","shap"]
    for d in data:
        for exp in exps:
            print(f"Running on {d} with {exp}:")
            ret = explain(d, exp)
    # breakpoint()
