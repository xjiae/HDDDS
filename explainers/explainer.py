import torch
from abc import ABC, abstractmethod


class Explainer(ABC):
    """
    Abstract class to implement custom explanation methods for a given.
    Parameters
    ----------
    model: any model implemented in pytorch
        Classifier we wish to explain.
    Methods
    -------
    get_explanations:
        Generate explanations for given input.
    Returns
    -------
    None
    """

    def __init__(self):
        # self.model = model
        pass

    # @abstractmethod
    # def get_explanation(self, model: torch.nn, inputs: torch.tensor):
    #     """
    #     Generate explanations for given input/s.
    #     Parameters
    #     ----------
    #     inputs: torch.tensor
    #         Input in two-dimensional shape (m, n).
    #     label: torch.tensor
    #         Label
    #     Returns
    #     -------
    #     torch.tensor
    #         Explanation vector/matrix.
    #     """
    #     pass
    