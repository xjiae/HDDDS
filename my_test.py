import torch
import torch.nn as nn
import numpy as np

from my_openxai import Explainer

# from explainers import *
from models import *
from hddds import *


snet = SimpleNet(in_shape=(3,24,24), out_shape=(10,))


x_to_explain = torch.rand(3,24,24)
x_train = torch.rand(100,3,24,24)

grad_explainer = Explainer(method="grad", model=snet)
intg_explainer = Explainer(method="ig", model=snet, dataset_tensor=torch.zeros(1,3,24,24))
lime_explainer = Explainer(method="lime", model=snet, dataset_tensor=x_train)
shap_explainer = Explainer(method="shap", model=snet, dataset_tensor=x_train)

n_test = 3
x_test = torch.rand(n_test, 3,24,24)  # We actually only care about w_test = torch.ones(1,3,24,24)
lbl_test = torch.randint(0,10, (n_test,))

grad_alpha = grad_explainer.get_explanation(x_test, lbl_test)   # Vanilla gradient
print(f"grad {grad_alpha.shape}")

intg_alpha = intg_explainer.get_explanation(x_test, lbl_test)   # Integrated gradients
print(f"intg {intg_alpha.shape}")

lime_alpha = lime_explainer.get_explanation(x_test, lbl_test)   # LIME
print(f"lime {lime_alpha.shape}")

shap_alpha = shap_explainer.get_explanation(x_test, lbl_test)   # SHAP
print(f"shap {shap_alpha.shape}")

