import os
import sys
import argparse

import torch

from datasets import *
from experiments import *

torch.manual_seed(1234)

# It is recommended that you just call these
def train_all_sample_models():
    train_sample_model("hai")
    train_sample_model("swat")
    train_sample_model("wadi")
    train_sample_model("mvtec")
    train_sample_model("squad")
  
def generate_all_sample_explanations(seeds=range(25), num_todo=100):
    generate_explanations_sample_mvtec(seeds, num_todo=num_todo)
    generate_explanations_sample_timeseries("swat", seeds, num_todo=num_todo)
    generate_explanations_sample_timeseries("hai", seeds, num_todo=num_todo)
    generate_explanations_sample_timeseries("wadi", seeds, num_todo=num_todo)
    generate_explanations_sample_squad(seeds, num_todo=num_todo)
    

def evaluate_all():
    timeseries()
    mvtec()
    squad()
    
if __name__ == "__main__":
    train_all_sample_models()
    generate_all_sample_explanations()
    evaluate_all()
    


