

import warnings
warnings.filterwarnings("ignore")
from run_model import *
from explainers.shap import *
from explainers.lime import LIME
from explainers.intgrad import IntGrad, IntegratedGradients
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from utils import *
import argparse
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.model_selection import train_test_split
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds_name", type=str, default="hai_all")
    parser.add_argument("-model", type=str, default="LR")
    parser.add_argument("-explainer", type=str, default="SHAP")
    args, unknown = parser.parse_known_args()
    return args
    

def main(args):
   
    # sample usage
    
    
    # load dataset
    BATCH_SIZE = 2024
    ds_name = args.ds_name
    explainer = args.explainer
    model_name = args.model
    
    
    dataset = get_dataset(ds_name)
    labels = dataset.label

    # Split the dataset into train and test sets
    # Perform stratified train-test split on the labels
    indices_train, indices_test = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels, random_state=42)

   # Create Subset objects for train and test sets
    train_dataset = torch.utils.data.Subset(dataset, indices_train)
    test_dataset = torch.utils.data.Subset(dataset, indices_test)
    
    # Create train and test dataloaders

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
         
    # model
    if model_name == "SVM":
        model = sklearn.svm.SVC(kernel="rbf", probability=True)
    elif model_name == "LR":
        model = sklearn.linear_model.LogisticRegression(random_state=0)
    scaler = MinMaxScaler()

    print("training....")
    for batch in tqdm(train_dataloader):
        x, y, a = batch
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        scaler.fit(x)
        model.fit(x, y)
        # break
  
    print("testing....")
    # explainer
    if explainer == 'LIME':
        exp_model = LIME(model)
    elif explainer == 'SHAP':
        exp_model = SHAP(model)
    elif explainer == 'INTG':
        exp_model = IntGrad(model)    
    alpha_pred = []
    alpha_true = []
    y_preds = []
    y_trues= []
    for batch in tqdm(test_dataloader):
        x, y, a = batch
        x = x.cpu().numpy()
        x = scaler.transform(x)
        y = y.cpu().numpy()
        a_pred = exp_model.fit(x, y)
        y_pred = model.predict(x)
        y_trues.append(y)
        y_preds.append(y_pred)
    
        if len(a_pred) > 0:
            a_true = a.cpu().numpy()[exp_model.ano_idx]
            alpha_pred.append(a_pred)
            alpha_true.append(a_true)
            # breakpoint()
        # if len(alpha_pred) > 1:
        #     break
 
        

    # evaluate the detection
    acc, f1, fpr, fnr = summary(np.hstack(y_trues).flatten(), np.hstack(y_preds).flatten(), score=False)
    
    f = open(f"results/{ds_name}_experiment_result_model.txt", "a")
    f.write(f"{model_name} & {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\ \n")
    f.close()
    
    # evaluate the explanation
    acc, f1, fpr, fnr = summary(np.vstack(alpha_true).flatten(), np.vstack(alpha_pred).flatten())
    
    f = open(f"results/{ds_name}_experiment_result_explainer.txt", "a")
    f.write(f"{model_name} | {explainer} & {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\ \n")
    f.close()
    

        

if __name__ == "__main__":
    args = parse_args()
    main(args)