import torch

from models import *
from train import *
from get_explanations import *


mvtec_dataset = MVTecDataset("bottle")

simple_net = SimpleNet(in_shape=(3,256,256), out_shape=(2,))
ffres = MyFastResA(return_mode="two_class")

grad_configs = GradConfigs()
intg_configs = IntGradConfigs()
lime_configs = LimeConfigs(x_train=torch.rand(100,3,256,256))
shap_configs = ShapConfigs()

dummy_grad_saveto = "saved_explanations/grad_dummy.pt"
dummy_intg_saveto = "saved_explanations/intg_dummy.pt"
dummy_lime_saveto = "saved_explanations/lime_dummy.pt"
dummy_shap_saveto = "saved_explanations/shap_dummy.pt"

grad_stuff = get_mvtec_explanation(simple_net, mvtec_dataset, grad_configs, saveto=dummy_grad_saveto, num_todo=50)

