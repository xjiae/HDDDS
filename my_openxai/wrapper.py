# Utils
import torch
import numpy as np

# Explanation Models
from .explainers import Gradient
from .explainers import IntegratedGradients
from .explainers import InputTimesGradient
from .explainers import SmoothGrad
from .explainers import LIME
from .explainers import SHAPExplainerC
from .explainers import RandomBaseline


def Explainer(method: str,
              model,
              dataset_tensor=None,
              param_dict_grad=None,
              param_dict_sg=None,
              param_dict_ig=None,
              param_dict_lime=None,
              param_dict_shap=None):
    
    if method == 'grad':
        if param_dict_grad is None:
            param_dict_grad = dict()
            param_dict_grad['absolute_value'] = True
        explainer = Gradient(model,
                             absolute_value=param_dict_grad['absolute_value'])
    
    elif method == 'sg':
        if param_dict_sg is None:
            param_dict_sg = dict()
            param_dict_sg['n_samples'] = 100
            param_dict_sg['standard_deviation'] = 0.005
        explainer = SmoothGrad(model,
                               num_samples=param_dict_sg['n_samples'],
                               standard_deviation=param_dict_sg['standard_deviation'])
    
    elif method == 'itg':
        explainer = InputTimesGradient(model)
    
    elif method == 'ig':
        if param_dict_ig is None:
            param_dict_ig = dict()
            param_dict_ig['method'] = 'gausslegendre'
            param_dict_ig['multiply_by_inputs'] = False
            if isinstance(dataset_tensor, torch.Tensor):
              param_dict_ig['baseline'] = torch.mean(dataset_tensor, dim=0).reshape(1, -1).float()
            else:
              param_dict_ig['baseline'] = None  # This will use the zero scalar
        explainer = IntegratedGradients(model,
                                        method=param_dict_ig['method'],
                                        multiply_by_inputs=param_dict_ig['multiply_by_inputs'],
                                        baseline=param_dict_ig['baseline'])
    
    elif method == 'shap':
        if param_dict_shap is None:
            param_dict_shap = dict()
            param_dict_shap['subset_size'] = 500
        explainer = SHAPExplainerC(model,
                                   model_impl='torch',
                                   n_samples=param_dict_shap['subset_size'])

    elif method == 'lime':
        if param_dict_lime is None:
            param_dict_lime = dict()
            assert isinstance(dataset_tensor, torch.Tensor)
            param_dict_lime['dataset_tensor'] = dataset_tensor
            param_dict_lime['kernel_width'] = 0.75
            param_dict_lime['std'] = float(np.sqrt(0.05))
            param_dict_lime['mode'] = 'tabular'
            param_dict_lime['sample_around_instance'] = True
            param_dict_lime['n_samples'] = 1000
            param_dict_lime['discretize_continuous'] = False

        @torch.no_grad()
        def predict_fn(x_np, batch_size=32):  # Change batch size if you OOM
          device = next(model.parameters()).device
          splits = torch.split(torch.tensor(np.float32(x_np)), batch_size)
          y = []
          for sp_x in splits:
            sp_y = model(sp_x.to(device))
            y.append(sp_y)
          y = torch.cat(y, dim=0)
          return y.detach().cpu().numpy()

        explainer = LIME(predict_fn,
                         param_dict_lime['dataset_tensor'],
                         std=param_dict_lime['std'],
                         mode=param_dict_lime['mode'],
                         sample_around_instance=param_dict_lime['sample_around_instance'],
                         kernel_width=param_dict_lime['kernel_width'],
                         n_samples=param_dict_lime['n_samples'],
                         discretize_continuous=param_dict_lime['discretize_continuous'])

    elif method == 'control':
        explainer = RandomBaseline(model)
    
    else:
        raise NotImplementedError("This method has not been implemented, yet.")
    
    return explainer


