

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
# import shap
# from openxai import Explainer
def threshold_array(arr, threshold):
    # Create a copy of the input array
    transformed_arr = np.copy(arr)
    
    # Set values larger than the threshold to 1, and all others to 0
    transformed_arr[transformed_arr > threshold] = 1
    transformed_arr[transformed_arr <= threshold] = 0
    
    return transformed_arr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds_name", type=str, default="swat")
    parser.add_argument("-model", type=str, default="SVM")
    parser.add_argument("-explainer", type=str, default="LIME")
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
    # breakpoint()
    # Split the dataset into train and test sets
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create train and test dataloaders

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
    # train_dataset = get_dataset('hat_train')
    # test_dataset = get_dataset('hai_test')
    # train_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
     # model
    # model = OneClassSVM(gamma='auto')
    # model = sklearn.svm.SVC(kernel="rbf", probability=True)
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
        # a_pred = threshold_array(a_pred, np.quantile(a_pred.flatten(), 0.95))
    
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
    
    
    # breakpoint()
    
    # with open("models/LSTM/StackedLSTM.pt", "rb") as f:
    #         SAVED_MODEL = torch.load(f)

    # # load 
    # model = StackedLSTM()
    # new = {k.replace('module.',''):v for k,v in SAVED_MODEL["state"].items()}
    # model.load_state_dict(new)
    # 
    # model.eval().cuda()
    # model =  nn.DataParallel(model)
    
    # shap = GradShapExplainer(top_k_frac=0.25) 
    # a_pred = intgrad.get_explanation(x, y)
    # a_pred = intgrad.fit(x, y)

    # exp = shap.find_explanation(model, x)
    
    # breakpoint()    
    
    # e = shap.DeepExplainer(model, background)
    # shap_values = e.shap_values(test_images)

        

if __name__ == "__main__":
    args = parse_args()
    main(args)