import shap
import math
import random
import sklearn
import numpy as np
import math
import torch
import torch.nn as nn
import shap

import copy
from .explainer import Explainer
from .wrapper import Wrapper

from torch.autograd import grad as original_grad

def new_grad(*args, **kwargs):
    kwargs['allow_unused'] = True

    return original_grad(*args, **kwargs)

# torch.autograd.grad = new_grad

class GradShapExplainer(Explainer):
    def __init__(self, top_k_frac, num_trains=100, num_samples=48, shap_batch_size=4):
        super(GradShapExplainer, self).__init__()
        self.top_k_frac = top_k_frac
        self.num_trains = num_trains
        self.num_samples = num_samples
        self.shap_batch_size = shap_batch_size

    def __str__(self):
        return f"shap{self.top_k_frac:.4f}"

    def find_explanation(self, model, x, get_shap_values=False, get_order=False, **kwargs):
        model.eval()
       
        # xx = x.unsqueeze(0)
        xx = x
        # xx_ashape = model.alpha_shape(xx)
        xx_ashape = xx.shape[-2:]
        
        alpha_train = torch.randint(0,2,(self.num_trains,*xx_ashape[1:]))
        alpha_train[0] = torch.ones(*xx_ashape[1:])

        cuda_model = Wrapper(model, x)
        breakpoint()
        explainer = shap.GradientExplainer(cuda_model, [alpha_train], batch_size=self.shap_batch_size)
        # explainer = shap.GradientExplainer(cuda_model, list(x), batch_size=self.shap_batch_size)
        breakpoint()
        raw_shap_values, _ = explainer.shap_values([torch.ones((1,xx_ashape[1]))], ranked_outputs=1, nsamples=self.num_samples)
        shap_values = torch.tensor(raw_shap_values[0][0])

        # The explanation
        k = math.ceil(shap_values.numel() * self.top_k_frac)
        order = shap_values.argsort(descending=True)
        exbits = torch.zeros(xx_ashape[1:]).to(x.device)
        exbits[order[:k]] = 1.0
        
        if get_shap_values:
            return exbits, raw_shap_values

        if get_order:
            return exbits, order

        return exbits


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

    def fit(self, x, y):

        self.dim = x.shape[1]


        self.ano_idx = np.where(y == 1)[0]

        # use Kernel SHAP to explain test set predictions
        # As instructed by SHAP, Using many background data samples could cause slower run times.
        # we use shap.kmeans(data, K) to summarize the background as 100 samples.

        x_kmean = shap.kmeans(x, self.n_sample)
        explainer = shap.KernelExplainer(self.model.predict_proba, x_kmean, link="logit")
        anomaly_shap_values = explainer.shap_values(x[self.ano_idx], nsamples="auto")

        anomaly_shap_values = anomaly_shap_values[1]
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