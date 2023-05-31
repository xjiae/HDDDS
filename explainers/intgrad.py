import sklearn
import numpy as np
from alibi.explainers import IntegratedGradients, Counterfactual
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import optimizers
import torch.nn.functional as F
import torch
# import onnx
# import onnx_tf


import torch

from captum.attr import IntegratedGradients as IG_Captum
import torch
import math

from .explainer import Explainer


# Vanilla gradient saliency
class PlainGradExplainer(Explainer):
  def __init__(self, top_k_frac, signed=False):
    super(PlainGradExplainer, self).__init__()
    self.top_k_frac = top_k_frac
    self.signed = signed

  def __str__(self):
    if self.signed:
      return f"pgrads{self.top_k_frac:.4f}"
    else:
      return f"pgradu{self.top_k_frac:.4f}"

  def _find_grad_order(self, model, x):
    # model.eval()
    xx = x.unsqueeze(0)
    test = torch.ones_like(x)
    test.requires_grad_()
    y = model(xx, alpha=test.unsqueeze(0))[0]
  
    
    v = y.max()
    return y, v, test
    # v.backward()
    
    # grad = test.grad.view(-1)
    # order = grad.argsort(descending=True) if self.signed else grad.abs().argsort(descending=True)
    # return grad.view(test.shape), order.view(test.shape)

  def find_explanation(self, model, x, get_order=False):
    model.eval()
    _, order = self._find_grad_order(model, x)
    
    # alpha_shape = model.alpha_shape(x.unsqueeze(0))[1:]
    k = math.ceil(order.numel() * self.top_k_frac)
    exbits = torch.zeros(order.numel()).to(x.device)
    exbits[order.view(-1)[:k]] = 1.0
    if get_order:
      return exbits.view(*x.shape), order
    else:
      return exbits.view(*x.shape)

# Integrated gradient saliency
class IntGradExplainer(Explainer):
  def __init__(self, top_k_frac, signed=False, num_steps=32):
    super(IntGradExplainer, self).__init__()
    self.top_k_frac = top_k_frac
    self.signed = signed
    self.num_steps = num_steps

  def __str__(self):
    if self.signed:
      return f"igrads{self.top_k_frac:.4f}"
    else:
      return f"igradu{self.top_k_frac:.4f}"

  def _intgrad(self, model, x, num_steps=None):
    num_steps = self.num_steps if num_steps is None else num_steps
    xx = x.unsqueeze(0)
    breakpoint()
    y = model(xx, torch.ones_like(xx))
    target_class = y.argmax(dim=1)[0]
    
    # alpha_shape = model.alpha_shape(xx)

    intgrad = torch.zeros(xx.shape).to(x.device)
    exbits_start = torch.zeros(xx.shape).to(x.device)
    exbits_final = torch.ones(xx.shape).to(x.device)

    for k in range(num_steps):
      exbits_this = exbits_start + (k/num_steps) * exbits_final
      exbits_this.requires_grad_()
      y_this = model(xx, alpha=exbits_this)
      y_target = y_this[0, target_class]
      y_target.backward()
      intgrad += (exbits_this.grad / num_steps) * (exbits_final - exbits_start)
    return intgrad[0]
  
  def find_explanation(self, model, x, get_order=False):
    model.eval()
    intgrad = self._intgrad(model, x)

    alpha_shape = model.alpha_shape(x.unsqueeze(0))[1:]
    tmp = intgrad.view(-1)
    order = tmp.argsort(descending=True) if self.signed else tmp.abs().argsort(descending=True)
    k = math.ceil(order.numel() * self.top_k_frac)
    exbits = torch.zeros(order.numel()).to(x.device)
    exbits[order.view(-1)[:k]] = 1.0
    if get_order:
      return exbits.view(*alpha_shape), order
    else:
      return exbits.view(*alpha_shape)





class IntegratedGradients:

    def __init__(self, model, method: str = 'gausslegendre', multiply_by_inputs: bool = False, baseline=None):
   

        self.method = method
        self.multiply_by_inputs = multiply_by_inputs
        self.baseline = baseline
        self.model = model

        # super(IntegratedGradients, self).__init__(model)

    def get_explanation(self, x: torch.Tensor, label: torch.Tensor):
        self.model.eval()
        self.model.zero_grad()

        ig = IG_Captum(self.model, self.multiply_by_inputs)

        attribution = ig.attribute(x, target=label, method=self.method, baselines=self.baseline)

        return attribution

class IntGrad:
    def __init__(self, model, n_steps=50, method="gausslegendre"):
        """

        :param n_steps:
        :param method:
        """
        self.clf_batch_size = 64
        self.clf_epochs = 30

        self.n_steps = n_steps
        self.method = method

        self.ano_idx = None
        self.nor_idx = None

        self.dim = None
        self.model = model
        return

    def fit(self, x, y):
        self.dim = x.shape[-1]
        # x = min_max_normalize(x)
        # # x = z_score_normalize(x)
        # y_oh = to_categorical(y, 2)
        # y_oh = F.one_hot(y)
        # clf = self.nn_model()
        # clf.fit(x, y_oh, batch_size=self.clf_batch_size, epochs=self.clf_epochs, verbose=1)
        # y_pred = clf(x).numpy().argmax(axis=1)
        # print("Clf model accuracy: [{:.4f}]".format(sklearn.metrics.accuracy_score(y, y_pred)))
        clf = self.model
        # Export the PyTorch model to ONNX
        # torch.onnx.export(self.model,               # model being run
        #                   torch.randn(1, 3, 224, 224), # dummy input (required)
        #                   "./models/LSTM/lstm.onnx",   # where to save the model (can be a file or file-like object)
        #                   export_params=True)
        # clf = onnx.load("./models/LSTM/lstm.onnx")
        # onnx.checker.check_model(clf)
        # clf = onnx_tf.convert_from_onnx(clf)
        
        # Initialize IntegratedGradients instance
        ig = IntegratedGradients(clf, method=self.method)
        shape = (1,) + x.shape[1:]
        cf = Counterfactual(self.model, shape, distance_fn='l1', target_proba=1.0,
                    target_class='other', max_iter=1000, early_stop=50, lam_init=1e-1,
                    max_lam_steps=10, tol=0.05, learning_rate_init=0.1,
                    feature_range=(-1e10, 1e10), eps=0.01, init='identity',
                    decay=True, write_dir=None, debug=False)

        # Calculate attributions for the first 10 images in the test set
        self.ano_idx = np.where(y == 1)[0]
        x_ano = x[self.ano_idx]
        # predictions = clf(x_ano).numpy().argmax(axis=1)
        predictions = np.ones(len(self.ano_idx), dtype=int)

        self.nor_idx = np.where(y == 0)[0]
        x_nor = x[self.nor_idx]
        x_nor_avg = np.average(x_nor, axis=0)
        baselines = np.array([x_nor_avg] * len(self.ano_idx))
        breakpoint()
        explanation = cf.explain(x_ano)
        explanation = ig.explain(x_ano, baselines=baselines, target=predictions)

        fea_weight_lst = explanation.data['attributions']
        return fea_weight_lst

    def nn_model(self):
        x_in = Input(shape=(self.dim,))
        x = Dense(10, activation='relu')(x_in)
        # x = Dense(10, activation='relu')(x)
        x_out = Dense(2, activation='softmax')(x)
        nn = Model(inputs=x_in, outputs=x_out)
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        nn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return nn


def min_max_normalize(x):
 
    n, dim = x.shape
    x_n = np.zeros(x.shape)
    for i in range(dim):
        array = x[:, i]
        _min, _max = np.min(array), np.max(array)
        if _min == _max:
            x_n[:, i] = np.zeros(n)
        else:
            x_n[:, i] = (array - _min) / (_max - _min)

    return x_n


def z_score_normalize(x):
    n, dim = x.shape
    x_n = np.zeros(x.shape)
    for i in range(dim):
        array = x[:, i]
        avg = np.average(array)
        std = np.std(array)
        if std != 0:
            x_n[:, i] = (array - avg) / std
        else:
            x_n[:, i] = array
    return x_n
