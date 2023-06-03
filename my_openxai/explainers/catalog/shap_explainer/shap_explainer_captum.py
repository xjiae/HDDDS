#import xgboost
#import shap
#import numpy as np
#from torch import nn
#from torch.nn import functional as F
#from shap import DeepExplainer
#from shap import KernelExplainer

import torch
from ...api import Explainer
from captum.attr import KernelShap

import warnings

class SHAPExplainerC(Explainer):
    '''
    param: model: model object
    param: data: pandas data frame or numpy array
    param: link: str, 'identity' or 'logit'
    param: feature_perturbation: str, 'tree_path_dependent' or 'interventional'
    '''

    def __init__(self, model, baseline_data: torch.FloatTensor = None,
                 model_impl: str = 'torch',
                 n_samples=500) -> None:
        super().__init__(model)
        self.n_samples = n_samples
        if model_impl == 'torch':
            self.explainer = KernelShap(model)
        elif model_impl == 'sklearn':
            self.explainer = KernelShap(self.forward_func_sklearn)

    def forward_func_sklearn(self, input):
        return torch.tensor(self.model.predict_proba(input))

    def forward_func_torch(self, input):
        return self.model(input)

    def get_explanation(self, data_x: torch.FloatTensor, label=None):
        '''
        Returns SHAP values as the explaination of the decision made for the input data (data_x)
        :param data_x: data samples to explain decision for
        :return: SHAP values [dim (shap_vals) == dim (data_x)]
        '''

        # Shut the fuck up
        warnings.filterwarnings("ignore")
        shap_vals = self.explainer.attribute(data_x, target=label, n_samples=self.n_samples)
        warnings.resetwarnings()
        return shap_vals.float()  # This one can handle both CPU and CUDA



