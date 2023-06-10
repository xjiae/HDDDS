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
from tqdm import tqdm
from datasets import *

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
    goods = MVTecDataset(root=mvtec_dir, category=category, input_size=256, is_train=True)
    anoms = MVTecDataset(root=mvtec_dir, category=category, input_size=256, is_train=False)
    concats = tudata.ConcatDataset([goods, anoms])
    # total = len(concats)
    # num_train = int(total * args.train_frac)
    # torch.manual_seed(1234 if seed is None else seed)
    # trains, vals = tudata.random_split(concats, [num_train, total - num_train])
    # train_loader = tudata.DataLoader(trains, batch_size=args.batch_size, shuffle=True)
    # val_loader = tudata.DataLoader(vals, batch_size=args.batch_size, shuffle=True)
    # return trains, vals, train_loader, val_loader
    return concats

def load_swat_data(ds_name):
    
    raw = 'raw' in ds_name
    
    sliding = 'sliding' in ds_name
    contents = None
    if 'train' in ds_name:
        contents = 'train'
    elif 'valid' in ds_name:
        contents = 'valid'
    else:
        contents = "all"
    if sliding:
        window_size = get_window(ds_name)
        return SWaTSlidingDataset(window_size, contents=contents)
    else:
        return SWaTDataset(contents=contents, raw=raw)
    
def load_wadi_data(ds_name):
    contents = None
    if 'train' in ds_name:
        contents = 'train'
    elif 'valid' in ds_name:
        contents = 'valid'
    else:
        contents = "all"
    raw = 'raw' in ds_name
    
    sliding = 'sliding' in ds_name
    if sliding:
        window_size = get_window(ds_name)
        return WADISlidingDataset(window_size, contents=contents)
    else:
        return WADIDataset(contents=contents, raw=raw)

def load_hai_data(ds_name):

    contents = None
    if 'train' in ds_name:
        contents = 'train'
    elif 'valid' in ds_name:
        contents = 'valid'
    else:
        contents = "all"
            
    raw = 'raw' in ds_name
  
    sliding = 'sliding' in ds_name
    
    if sliding:
        window_size = get_window(ds_name)
        return HAISlidingDataset(window_size, contents=contents)
    else:
        return HAIDataset(contents=contents, raw=raw)
       
def get_window(ds_name):
    digit_str = re.findall(r'\d+', ds_name)
    digit = int(digit_str[0]) if digit_str else None
    return digit

def load_squad_data(ds_name):
    train = "train" in ds_name
    squad = SquadDataset("roberta-base", is_train=train)
    return squad

def get_dataset(ds_name):
    if 'mvtec' in ds_name:
        return load_mvtec_data(ds_name.split('_')[1])
    elif 'swat' in ds_name:
        return load_swat_data(ds_name)
    elif 'wadi' in ds_name:
        return load_wadi_data(ds_name)
    elif 'hai' in ds_name:
        return load_hai_data(ds_name)
    elif 'squad' in ds_name:
        return load_squad_data(ds_name)

if __name__ == "__main__":
    # # example
    # dataset = get_dataset('mvtec_bottle')
    # example
    # dataset = get_dataset('swat')
    # example
    # dataset = get_dataset('wadi')
    dataset = get_dataset('squad_train')
    breakpoint()
    


# dictionary for DSInfo
