import torch

from models import *
from train import *
from get_explanations import *
from hddds import *
from utils import *
from dataset import *
import os
import sys
sys.path.insert(0, "/home/xjiae/HDDDS/")
def evaluate(model, ds_name, exp):
    
    loadfrom = open(f"saved_explanations/{model}_{exp}_{ds_name}.pt", "rb")
    ret = torch.load(loadfrom)
    w_true = torch.cat(ret['w_s']).numpy().flatten()
    w_pred = torch.cat(ret['w_exps']).numpy().flatten()
    acc, f1, fpr, fnr = summary(w_true, w_pred, score = True)
    saveto = open(f"results/{ds_name}.txt", "a")
    saveto.write(f"{model} | {exp} & {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\ \n")
    saveto.close()
    
    
def explain(model_name, ds_name, exp):
    
    # ret = get_hai_dataloaders()
    # data = ret['test_dataset']
    data = get_dataset(ds_name)
    
    config = None
    saveto = os.path.join("saved_explanations", f"{model_name}_{exp}_{ds_name}.pt")
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
    if model_name == "lstm":
        model = SimpleLSTM(in_shape=(data.num_features,), out_shape=(1,), return_mode=ret_mode)
    else:
        model = LogisticRegression(in_shape=(86,), out_shape=(1,), return_mode=ret_mode)
    # mp = ds_name.replace("test", "")
    mp = ds_name.split("_")[0]
    if model_name == "lstm":
        model_path = open(f"saved_models/{mp}-sliding_epoch2.pt", "rb") 
    else:
        model_path = open(f"saved_models/lr_{mp}_epoch1.pt", "rb") 
        
    load = torch.load(model_path)
    model.load_state_dict(load)
    # ret_dict = get_tabular_sliding_explanation(model, data, config, saveto=saveto, device=device, num_todo=1000)
    ret_dict = get_tabular_explanation(model, data, config, saveto=saveto, device=device, num_todo=1000)

    return ret_dict

if __name__== "__main__":

    data = ["swat_sliding_100_valid", "wadi_sliding_100_valid", "hai_sliding_100_valid"]
    exps = ["grad","intg","lime"]
    # exps = ["shap","lime"]
    data = ["hai_100_test"]
    exps = ["lime"]
    m = "lr"
    for d in data:
        for exp in exps:
            print(f"Running {m} on {d} with {exp}:")
            # ret = explain(m, d, exp)
            evaluate(m, d, exp)
    # breakpoint()
