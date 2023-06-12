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

from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor
import tensorflow_datasets as tfds

import torch.utils.data as tud

# Adapted from: https://github.com/TheAtticusProject/cuad/blob/main/train.py
# Light wrapper around a TensorDataset where elements are 5-tuples of:
#   index 0: input_ids
#   index 1: attention_masks
#   index 2: token_type_ids
#   index 3: start_positions
#   index 4: end_positions
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

      self.max_seq_len = max_seq_len
      self.max_query_len = max_query_len
      self.doc_stride = doc_stride
      self.is_train = is_train

      # Set up caching information
      self.cache_dir = cache_dir
      if not os.path.exists(self.cache_dir):
          os.makedirs(self.cache_dir)

      self._train_cache_file = os.path.join(cache_dir, f"train_{self.tokenizer_name}_{max_seq_len}.cache")
      self._test_cache_file = os.path.join(cache_dir, f"test_{self.tokenizer_name}_{max_seq_len}.cache")
      self.cache_file = self._train_cache_file if is_train else self._test_cache_file

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
        input_ids = item[0]
        attention_mask = item[1]
        token_type_ids = item[2]
        start_position = item[3]
        end_position = item[4]
        if self.is_train:
            return input_ids, \
                   attention_mask, \
                   token_type_ids, \
                   start_position, \
                   end_position
        else:
            return input_ids, \
                   attention_mask, \
                   token_type_ids
    
    def __len__(self):
        return len(self.dataset)


def load_dataset(tokenizer, is_train, max_seq_len, max_query_len, doc_stride):
    # processor = SquadV1Processor()
    processor = SquadV2Processor()
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

    return dataset

# The squad bundle
def get_squad_bundle(tokenizer_or_name = "roberta-base",
                     train_batch_size = 16,
                     test_batch_size = 16,
                     shuffle = True,
                     **kwargs):
  trains = SquadDataset(tokenizer_or_name, is_train=True, **kwargs)
  tests = SquadDataset(tokenizer_or_name, is_train=False, **kwargs)
  tokenizer = trains.tokenizer

  trains_perm = torch.randperm(len(trains)) if shuffle else torch.tensor(range(len(trains)))
  tests_perm = torch.randperm(len(tests)) if shuffle else torch.tensor(range(len(tests)))
  trains = tud.Subset(trains, indices=trains_perm)
  tests = tud.Subset(tests, indices=tests_perm)

  train_dataloader = tud.DataLoader(trains, batch_size=train_batch_size, shuffle=shuffle)
  test_dataloader = tud.DataLoader(tests, batch_size=test_batch_size, shuffle=shuffle)
  return { "train_dataset" : trains,
           "test_dataset" : tests,
           "train_dataloader" : train_dataloader,
           "test_dataloader" : test_dataloader,
           "tokenizer" : tokenizer,
         }


