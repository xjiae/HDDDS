from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
def threshold_binary(row, threshold):
    return np.where(row >= threshold, 1, 0)
def threshold_array(arr, threshold):
    # Create a copy of the input array
    transformed_arr = np.copy(arr)
    
    # Set values larger than the threshold to 1, and all others to 0
    transformed_arr[transformed_arr > threshold] = 1
    transformed_arr[transformed_arr <= threshold] = 0
    
    return transformed_arr
def summary(y_true, y_pred, score = True):
  
    if score:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2*recall*precision/(recall+precision)
        # weights = confusion_matrix(y_true, y_pred).sum(axis=1)
        # weighted_f1_scores = np.average(f1_scores, weights=weights)
        threshold = thresholds[np.argmax(f1_scores)]
        
        
        y_pred = threshold_array(y_pred, threshold)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    # print("-"*100)
    # print(f"fpr: {fpr:.2f}")
    # print(f"fnr: {fnr:.2f}")
    # print(f"acc: {acc:.2f}")
    # print(f"f1: {f1:.2f}")
    # print("-"*100)
    print(f"& {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\")
    return acc, f1, fpr, fnr