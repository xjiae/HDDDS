import os
import sys
import copy
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tudata
print(torch.__version__)
# from tqdm import tqdm

# Our stuff

from mvtec import MVTecDataset
from swat import SWaTDataset
from wadi import WADIDataset
from hai import HAIDataset

torch.manual_seed(1234)
#####
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
mvtec_dir = 'data/mvtec-ad'

# Returns the train and validation dataset
def load_mvtec_data(category, seed=None):
    
    goods = MVTecDataset(mvtec_dir, category, 256, is_train=True)
    anoms = MVTecDataset(mvtec_dir, category, 256, is_train=False)
    concats = tudata.ConcatDataset([goods, anoms])
    # total = len(concats)
    # num_train = int(total * args.train_frac)
    # torch.manual_seed(1234 if seed is None else seed)
    # trains, vals = tudata.random_split(concats, [num_train, total - num_train])
    # train_loader = tudata.DataLoader(trains, batch_size=args.batch_size, shuffle=True)
    # val_loader = tudata.DataLoader(vals, batch_size=args.batch_size, shuffle=True)
    # return trains, vals, train_loader, val_loader
    return concats

def load_swat_data():
    swat = SWaTDataset()
    return swat
def load_wadi_data():
    wadi = WADIDataset()
    return wadi
def load_hai_data():
    hai = HAIDataset()
    return hai
def get_dataset(ds_name):
    if 'mvtec' in ds_name:
        return load_mvtec_data(ds_name.split('_')[1])
    elif 'swat' in ds_name:
        return load_swat_data()
    elif 'wadi' in ds_name:
        return load_wadi_data()
    elif 'hai' in ds_name:
        return load_hai_data()

if __name__ == "__main__":
    # # example
    # dataset = get_dataset('mvtec_bottle')
    # example
    # dataset = get_dataset('swat')
    # example
    # dataset = get_dataset('wadi')
    dataset = get_dataset('hai')
    breakpoint()
    


# dictionary for DSInfo
