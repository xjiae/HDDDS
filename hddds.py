import os
import sys
import argparse

import torch

from datasets import *
from train import *
from get_explanations import *

torch.manual_seed(1234)

# It is recommended that you just call these
def train_all_sample_models():
    train_sample_model("hai")
    train_sample_model("swat")
    train_sample_model("wadi")
    train_sample_model("mvtec")
    train_sample_model("squad")
  
def generate_all_sample_explanations(seeds=range(50), num_todo=100):
    generate_explanations_sample_mvtec(seeds, num_todo=num_todo)
    generate_explanations_sample_timeseries("swat", seeds, num_todo=num_todo)
    generate_explanations_sample_timeseries("hai", seeds, num_todo=num_todo)
    generate_explanations_sample_timeseries("wadi", seeds, num_todo=num_todo)
    


