import torch

from models import *
from train import *
from get_explanations import *
from hddds import *
from utils import *
import os
import sys
sys.path.insert(0, "/home/xjiae/HDDDS/")
def evaluate(ds_name, exp):
    loadfrom = open(f"saved_explanations/{exp}_{ds_name}.pt", "rb")
    ret = torch.load(loadfrom)
    w_true = torch.cat(ret['w_s']).numpy().flatten()
    w_pred = torch.cat(ret['w_exps']).numpy().flatten()
    acc, f1, fpr, fnr = summary(w_true, w_pred, score = True)
    saveto = open(f"results/{ds_name}.txt", "a")
    saveto.write(f"LSTM | {exp} & {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\ \n")
    saveto.close()
    
    
def explain(ds_name, exp):
    
    data = get_dataset(ds_name)
    
    config = None
    saveto = os.path.join("saved_explanations", f"{exp}_{ds_name}.pt")
    ret_mode = "two_class"
    device = "cuda"
    
    match exp:
        case "grad":
            # for lstm, need to be in train mode
            config = GradConfigs(train_mode=True)
           
        case "intg":
            # for lstm, need to be in train mode
            config = IntGradConfigs(train_mode=True)
            
        case "lime":
            config = LimeConfigs(x_train=torch.rand(100,*data[0][0].shape))
            
        case "shap":
            config = ShapConfigs()
            
            device = "cpu"
    model = SimpleLSTM(in_shape=(data.num_features,), out_shape=(1,), return_mode=ret_mode)
    mp = ds_name.replace("test", "train")
    model_path = open(f"saved_models/{mp}_epoch1.pt", "rb") 
    load = torch.load(model_path)
    model.load_state_dict(load)
    ret_dict = get_tabular_explanation(model, data, config, saveto=saveto, device=device, num_todo=1000)

    return ret_dict

if __name__== "__main__":

    data = ["hai_sliding_100_test","swat_sliding_100_test", "wadi_sliding_100_test"]
    exps = ["grad","intg","lime","shap"]
    data = ["hai_sliding_100_test"]
    exps = ["shap"]
    for d in data:
        for exp in exps:
            print(f"Running on {d} with {exp}:")
            ret = explain(d, exp)
            evaluate(d, exp)
    # breakpoint()
