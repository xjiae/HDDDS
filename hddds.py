import os
import sys
import argparse

from datasets import *
from train import *
from get_explanations import *

# It is recommended that you just call these
def train_all_sample_models():
    train_sample_model("hai")
    train_sample_model("swat")
    train_sample_model("wadi")
    train_sample_model("mvtec")
    train_sample_model("squad")
  



