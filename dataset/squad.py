import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformers
from transformers import (
    AutoTokenizer,
    squad_convert_examples_to_features,
)

from transformers.data.processors.squad import SquadV1Processor
import tensorflow_datasets as tfds

import torch.utils.data as tud

# Adapted from: https://github.com/TheAtticusProject/cuad/blob/main/train.py
# Light wrapper around a TensorDataset where elements are 8-tuples of:
#   index 0: input_ids
#   index 1: attention_masks
#   index 2: token_type_ids
#   index 3: start_positions
#   index 4: end_positions
#   index 5: cls_index
#   index 6: p_mask
#   index 7: is_impossible
class SquadDataset(tud.Dataset):
  def __init__(self,
               tokenizer_or_name, # Need to supply the name of the classifier
               tokenizer_name = "defaultname", # Gets overriden if tokenizer_or_name is str 
               max_seq_len = 384,
               max_query_len = 64,
               doc_stride = 128,
               cache_dir = "data/squad/cache",
               overwrite_cache = False,
               is_train = True,
               seed = 1234):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up the tokenizer
    if isinstance(tokenizer_or_name, str):
      # Squad processing doesn't work with fast mode
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_name, use_fast=False)
      self.tokenizer_name = tokenizer_or_name
    else:
      self.tokenizer = tokenizer_or_name
      self.tokenizer_name = tokenizer_name

    self.max_seq_len= max_seq_len
    self.max_query_len= max_query_len
    self.doc_stride = doc_stride
    self.is_train = is_train

    # Set up caching information
    self.cache_dir = cache_dir
    if not os.path.exists(self.cache_dir):
      os.makedirs(self.cache_dir)

    self._train_cache_file = os.path.join(cache_dir, f"train_{self.tokenizer_name}_{max_seq_len}.cache")
    self._eval_cache_file = os.path.join(cache_dir, f"eval_{self.tokenizer_name}_{max_seq_len}.cache")
    self.cache_file = self._train_cache_file if is_train else self._eval_cache_file

    # Use the cache if okay
    if os.path.exists(self.cache_file) and not overwrite_cache:
      print(f"loading cache from {self.cache_file}")
      self.dataset = torch.load(self.cache_file)
    else:
      self.dataset = load_dataset(self.tokenizer,
                                  is_train,
                                  max_seq_len,
                                  max_query_len,
                                  doc_stride)
      torch.save(self.dataset, self.cache_file)
      print(f"cached to {self.cache_file}")

  def __getitem__(self, idx):
    item = self.dataset[idx]

    '''
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        "start_positions": batch[3],
        "end_positions": batch[4],
    }]

    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
    }
    '''
    return item

  
  def __len__(self):
    return len(self.dataset)


def load_dataset(tokenizer, is_train, max_seq_len, max_query_len, doc_stride):
    processor = SquadV1Processor()
    '''
    if is_train:
        examples = processor.get_train_examples(data_dir, filename=filename)
    else:
        examples = processor.get_dev_examples(data_dir, filename=filename)
    '''
    tfds_examples = tfds.load("squad")
    examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=not is_train)
    features, dataset = squad_convert_examples_to_features(
        examples = examples,
        tokenizer = tokenizer,
        max_seq_length = max_seq_len,
        doc_stride = doc_stride,
        max_query_length = max_query_len,
        is_training = is_train,
        return_dataset = "pt")

    if is_train:
      return get_balanced_dataset(dataset)
    else:
      return dataset


def get_balanced_dataset(dataset):
    """
    returns a new dataset, where positive and negative examples are approximately balanced
    """
    pos_mask = get_dataset_pos_mask(dataset)
    neg_mask = [~mask for mask in pos_mask]
    npos, nneg = np.sum(pos_mask), np.sum(neg_mask)

    neg_keep_frac = npos / nneg  # So that in expectation there will be npos negative examples (--> balanced)
    neg_keep_mask = [mask and np.random.random() < neg_keep_frac for mask in neg_mask]

    # keep all positive examples and subset of negative examples
    keep_mask = [pos_mask[i] or neg_keep_mask[i] for i in range(len(pos_mask))]
    keep_indices = [i for i in range(len(keep_mask)) if keep_mask[i]]

    subset_dataset = torch.utils.data.Subset(dataset, keep_indices)
    return subset_dataset


def get_dataset_pos_mask(dataset):
    """
    Returns a list, pos_mask, where pos_mask[i] indicates is True if the ith example in the dataset is positive
    (i.e. it contains some text that should be highlighted) and False otherwise.
    """
    pos_mask = []
    for i in range(len(dataset)):
        ex = dataset[i]
        start_pos = ex[3]
        end_pos = ex[4]
        is_positive = end_pos > start_pos
        pos_mask.append(is_positive)
    return pos_mask


def get_squad_dataloaders(tokenizer_or_name = "roberta-base",
                          train_batch_size = 8,
                          valid_batch_size = 8,
                          **kwargs):
  trains = SquadDataset(tokenizer_or_name, is_train=True, **kwargs)
  valids = SquadDataset(tokenizer_or_name, is_train=False, **kwargs)
  train_loader = tud.DataLoader(trains, batch_size=train_batch_size, shuffle=True)
  valid_loader = tud.DataLoader(valids, batch_size=valid_batch_size, shuffle=True)
  return { "train_dataset" : trains,
           "valid_dataset" : valids,
           "train_dataloader" : train_loader,
           "valid_dataloader" : valid_loader }


