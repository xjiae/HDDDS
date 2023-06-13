import os
import sys
import argparse

from datasets import *
from train import *
from get_explanations import *



'''
hai_bundle = get_data_bundle("hai", window_size=100)
hai_trains = hai_bundle["train_dataset"]
'''

# hai_all = TimeSeriesDataset("hai", 100, label_choice="all")
hai_last = TimeSeriesDataset("hai", 100, label_choice="last")
breakpoint()
