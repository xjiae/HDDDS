from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score

def threshold_binary(row, threshold):
    return np.where(row >= threshold, 1, 0)

def summary(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return acc, f1, fpr, fnr