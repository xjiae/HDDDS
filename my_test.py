import torch
import torch.nn as nn
import numpy as np

from my_openxai import Explainer

# from explainers import *
from models import *
from hddds import *


snet = SimpleNet(in_shape=(3,24,24), out_shape=(10,))



x_to_explain = torch.rand(3,24,24)

grad_explainer = Explainer(method="grad", model=XWrapper(snet, x_to_explain))
intg_explainer = Explainer(method="ig", model=XWrapper(snet, x_to_explain), dataset_tensor=torch.zeros(1,3,24,24))

# Lime and SHAP requires a bunch of sampled "training" points that we make-up below
w_train = torch.rand(100,3,24,24)
lime_explainer = Explainer(method="lime", model=XWrapper(snet, x_to_explain), dataset_tensor=w_train)
shap_explainer = Explainer(method="shap", model=XWrapper(snet, x_to_explain), dataset_tensor=w_train)

n_test = 5
w_test = torch.rand(n_test, 3,24,24)  # We actually only care about w_test = torch.ones(1,3,24,24)
lbl_test = torch.randint(0,10, (n_test,))

grad_alpha = grad_explainer.get_explanation(w_test, lbl_test)   # Vanilla gradient
intg_alpha = intg_explainer.get_explanation(w_test, lbl_test)   # Integrated gradients
lime_alpha = lime_explainer.get_explanation(w_test, lbl_test)   # LIME
shap_alpha = shap_explainer.get_explanation(w_test, lbl_test)   # SHAP

print(f"grad {grad_alpha.shape}")
print(f"intg {intg_alpha.shape}")
print(f"lime {lime_alpha.shape}")
print(f"shap {shap_alpha.shape}")

