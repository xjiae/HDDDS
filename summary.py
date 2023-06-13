import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve

PT_FOLDER = "saved_explanations"


def threshold_array(arr, threshold):
    # Create a copy of the input array
    transformed_arr = np.copy(arr)
    
    # Set values larger than the threshold to 1, and all others to 0
    transformed_arr[transformed_arr > threshold] = 1
    transformed_arr[transformed_arr <= threshold] = 0
    
    return transformed_arr
def summary(y_true, y_pred, score = True):
    y_true = y_true.astype(int)
    if score:

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        recall = recall + 1e-10
        f1_scores = 2*recall*precision/(recall+precision)
        # breakpoint()
        threshold = thresholds[np.argmax(f1_scores)]
        
        
        y_pred = threshold_array(y_pred, threshold)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    # print(f"& {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\")
    return acc, f1, fpr, fnr


def evaluate(ret):
  w_true = torch.cat(ret['ws']).numpy().flatten()
  w_pred = torch.cat(ret['w_exps']).numpy().flatten()
  acc, f1, fpr, fnr = summary(w_true, w_pred, score=True)
  return acc, f1, fpr, fnr 

def mean_std(lst):
    return np.mean(lst), np.std(lst)

def compute_stat(rets, model_name, ds_name, attr_name, latex=True):
    fpr, fpr_std = mean_std(rets['fpr'])
    fnr, fnr_std = mean_std(rets['fnr'])
    acc, acc_std = mean_std(rets['acc'])
    f1, f1_std = mean_std(rets['f1'])
    if latex:
        saveto = open(f"results/results_{ds_name}.txt", "a")
        saveto.write(f"{model_name} & {attr_name} & {fpr:.2f} $\pm$ {fpr_std:.2f} & {fnr:.2f} $\pm$ {fnr_std:.2f} & {acc:.2f} $\pm$ {acc_std:.2f} & {f1:.2f} $\pm$ {f1_std:.2f} \\\\\n")
        saveto.close()
    else:
        return [fpr, fpr_std, fnr, fnr_std, acc, acc_std, f1, f1_std]
        
    
def get_table(ds_name, attr):
    rets = defaultdict(list)
    for file in os.listdir(PT_FOLDER):
        if ds_name in file and attr in file:
            fp = os.path.join(PT_FOLDER, file)
            if os.path.getsize(fp) > 0:  
                print(file)
                f = open(fp, "rb")
                ret = torch.load(f)
                acc, f1, fpr, fnr = evaluate(ret)
                rets['fpr'].append(fpr)
                rets['fnr'].append(fnr)
                rets['acc'].append(acc)
                rets['f1'].append(f1)
    compute_stat(rets, "model", ds_name, attr)

def compute_overall_avg(ds_name, attr):
    rets = defaultdict(list)
    for file in os.listdir(PT_FOLDER):
        if ds_name in file and attr in file:
            fp = os.path.join(PT_FOLDER, file)
            if os.path.getsize(fp) > 0:  
                print(file)
                f = open(fp, "rb")
                ret = torch.load(f)
                acc, f1, fpr, fnr = evaluate(ret)
                rets['fpr'].append(fpr)
                rets['fnr'].append(fnr)
                rets['acc'].append(acc)
                rets['f1'].append(f1)
    ret = compute_stat(rets, None, ds_name, attr, False)    
    return ret    
    
    


def all():
    all = tabular + image + text
    # all = tabular
    
    for attr in attr_models:
        rets = defaultdict(list)
        for ds_name in all:
            fpr, fpr_std, fnr, fnr_std, acc, acc_std, f1, f1_std = compute_overall_avg(ds_name, attr)
            rets['fpr'].append(fpr)
            rets['fnr'].append(fnr)
            rets['acc'].append(acc)
            rets['f1'].append(f1)
            rets['fpr_std'].append(fpr_std)
            rets['fnr_std'].append(fnr_std)
            rets['acc_std'].append(acc_std)
            rets['f1_std'].append(f1_std)
            
        saveto = open(f"results/results_all.txt", "a")
        saveto.write(f"{attr}\n")
        saveto.write(f"fpr: {rets['fpr']}\n")
        saveto.write(f"fpr_std: {rets['fpr_std']}\n")
        saveto.write(f"fnr: {rets['fnr']}\n")
        saveto.write(f"fnr_std: {rets['fnr_std']}\n")
        saveto.write(f"acc: {rets['acc']}\n")
        saveto.write(f"acc_std: {rets['acc_std']}\n")
        saveto.write(f"f1: {rets['f1']}\n")
        saveto.write(f"f1_std: {rets['f1_std']}\n")
        saveto.write(f"fpr_avg: {np.mean(rets['fpr']):.4f}\n")
        saveto.write(f"fnr_avg: {np.mean(rets['fnr']):.4f}\n")
        saveto.write(f"acc_avg: {np.mean(rets['acc']):.4f}\n")
        saveto.write(f"f1_avg: {np.mean(rets['f1']):.4f}\n")
        saveto.close()



attr_models = ["lime","shap"]
# , "swat", "wadi"
tabular = ["hai", "swat", "wadi"]
image = ["mvtec"]
text = ["squad"]

tabular_models = ['lstm', 'lr']
image_models = ["ffres"]
text_models = ["roberta"]

def timeseries():
    for ds_name in tqdm(tabular):
        print(ds_name)
        # for model in tabular_models:
        for attr in attr_models:
            get_table(ds_name, attr)

def mvtec():
    for ds_name in image:
        for model in image_models:
            for attr in attr_models:
                print(f"running {attr}...")
                get_table(ds_name, attr)


def squad():
    for ds_name in text:
        for model in text_models:
            for attr in attr_models:
                print(f"running {attr}...")
                get_table(ds_name, attr)
                


