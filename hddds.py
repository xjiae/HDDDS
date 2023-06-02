import os
import sys
import copy
import random
import argparse
from pathlib import Path
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tudata
# from tqdm import tqdm

# Our stuff

from dataset.mvtec import MVTecDataset
from dataset.swat import SWaTDataset, SWaTSlidingDataset
from dataset.wadi import WADIDataset, WADISlidingDataset
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
mvtec_dir = '../mvtec-ad'

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

def load_swat_data(ds_name, sliding = False):
    train = 'train' in ds_name
    raw = 'raw' in ds_name
    all = 'all' in ds_name
    if sliding:
        return SWaTSlidingDataset(train = train)
    else:
        return SWaTDataset(all = all, train = train, raw=raw)
    
def load_wadi_data(ds_name, sliding = False):
    train = 'train' in ds_name
    raw = 'raw' in ds_name
    all = 'all' in ds_name
    if sliding:
        return WADISlidingDataset(train = train)
    else:
        return WADIDataset(all = all, train = train, raw=raw)

def load_hai_data(ds_name):
    train = 'train' in ds_name
    raw = 'raw' in ds_name
    all = 'all' in ds_name
    sliding = 'sliding' in ds_name
    
    if sliding:
        window_size = get_window(ds_name)
        return HAISlidingDataset(window_size, train = train)
    else:
        return HAIDataset(all = all, train = train, raw=raw)
       
def get_window(ds_name):
    digit_str = re.findall(r'\d+', ds_name)
    digit = int(digit_str[0]) if digit_str else None
    return digit

def load_cuad_data():
    cuad = None
    return cuad

def get_dataset(ds_name):
    if 'mvtec' in ds_name:
        return load_mvtec_data(ds_name.split('_')[1])
    elif 'swat' in ds_name:
        return load_swat_data(ds_name)
    elif 'wadi' in ds_name:
        return load_wadi_data(ds_name)
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
