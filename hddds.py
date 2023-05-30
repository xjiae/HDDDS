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

from dataset.mvtec import MVTecDataset
from dataset.swat import SWaTDataset
from dataset.wadi import WADIDataset
from dataset.hai import HAIDataset, HAISlidingDataset

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
def load_hai_data(ds_name):
    if 'train' in ds_name:
        return HAIDataset(train = True)
    elif 'test' in ds_name:
        # return HAISlidingDataset(train=False)
        return HAIDataset(train = False)
    return HAIDataset(all=True)
    # return HAISlidingDataset()

def load_cuad_data():
    cuad = None
    return cuad

def get_dataset(ds_name):
    if 'mvtec' in ds_name:
        return load_mvtec_data(ds_name.split('_')[1])
    elif 'swat' in ds_name:
        return load_swat_data()
    elif 'wadi' in ds_name:
        return load_wadi_data()
    elif 'hai' in ds_name:
        return load_hai_data(ds_name)
    elif 'cuad' in ds_name:
        return load_cuad_data()

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
