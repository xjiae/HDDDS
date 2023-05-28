import shap
import math
import random
import sklearn
import numpy as np


class SHAP:
    def __init__(self, model, kernel="rbf", n_sample=100, threshold=0.8):
        """

        :param kernel: clf model svm parameter
        :param threshold: threshold is used to filter feature subset for each data, the shap values of selected feature
        subspace accounts for [threshold] of the sum of the shap values of feature full space.
        """
        self.ano_idx = None

        self.kernel = kernel
        self.threshold = threshold
        self.n_sample = n_sample
        self.dim = None
        self.model = model
        self.explainer = None

        return

    def fit(self, x):

        self.dim = x.shape[1]


        # x_kmean = shap.kmeans(x, self.n_sample)
        breakpoint()
        self.explainer = shap.DeepExplainer(self.model, x[:self.n_sample])
        
        anomaly_shap_values = self.explainer.shap_values(x)
       

        # anomaly_shap_values = anomaly_shap_values[1]
        return anomaly_shap_values

    def weight2subspace(self, weight, r=0.7, num=-1):
        threshold = r * np.sum(weight)
        tmp_s = 0
        exp_subspace = []
        sorted_idx1 = np.argsort(weight)
        sorted_idx = [sorted_idx1[self.dim - i -1] for i in range(self.dim)]
        if num != -1:
            exp_subspace = sorted_idx[:num]
            exp_subspace = list(np.sort(exp_subspace))
            return exp_subspace

        for idx in sorted_idx:
            tmp_s += weight[idx]
            exp_subspace.append(idx)
            if tmp_s >= threshold:
                break
        exp_subspace = list(np.sort(exp_subspace))
        return exp_subspace

    def weight2subspace_pn(self, weight):
        exp_subspace = []
        for i in range(len(weight)):
            if weight[i] > 0:
                exp_subspace.append(i)
        exp_subspace = list(np.sort(exp_subspace))
        return exp_subspace

    def get_exp_subspace(self, fea_weight_lst, w2s_ratio, real_exp_len=None):
        exp_subspace_lst = []
        for ii, idx in enumerate(self.ano_idx):
            fea_weight = fea_weight_lst[ii]
            if w2s_ratio == "real_len":
                exp_subspace_lst.append(self.weight2subspace(fea_weight, num=real_exp_len[ii]))
            elif w2s_ratio == "auto":
                r = math.sqrt(2 / self.dim)
                exp_subspace_lst.append(self.weight2subspace(fea_weight, r=r))
            elif w2s_ratio == "pn":
                exp_subspace_lst.append(self.weight2subspace_pn(fea_weight))
            else:
                exp_subspace_lst.append(self.weight2subspace(fea_weight, r=w2s_ratio))
        return exp_subspace_lst