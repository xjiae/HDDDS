import os
import numpy as np
import torch
import torch.utils.data as tud

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    AutoTokenizer,
    squad_convert_examples_to_features,
)

from transformers.data.processors.squad import SquadResult, SquadV1Processor

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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
class CuadDataset(tud.Dataset):
  def __init__(self,
               tokenizer_or_name, # Need to supply the name of the classifier
               tokenizer_name = "default-desc", # Gets overriden if tokenizer_or_name is str 
               max_seq_len = 384,
               max_query_len = 64,
               doc_stride = 128,
               data_dir = "data/cuad",
               train_filename = "train.json",
               eval_filename = "test.json",
               cache_dir = "data/cuad/cache",
               overwrite_cache = False,
               is_train = True,
               seed = 1234):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make sure some things exist, and create if not
    self.data_dir = data_dir
    assert os.path.isdir(data_dir)

    self.train_filename = train_filename
    self.train_file = os.path.join(data_dir, train_filename)
    assert os.path.isfile(self.train_file)

    self.eval_filename = eval_filename
    self.eval_file = os.path.join(data_dir, eval_filename)
    assert os.path.isfile(self.eval_file)

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
      filename = train_filename if is_train else eval_filename
      self.dataset = load_cuad_dataset(self.tokenizer,
                                       data_dir,
                                       filename,
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


def load_cuad_dataset(tokenizer, data_dir, filename, is_train, max_seq_len, max_query_len, doc_stride):
    processor = SquadV1Processor()
    if is_train:
        examples = processor.get_train_examples(data_dir, filename=filename)
    else:
        examples = processor.get_dev_examples(data_dir, filename=filename)
    features, dataset = squad_convert_examples_to_features(
        examples = examples,
        tokenizer = tokenizer,
        max_seq_length = max_seq_len,
        doc_stride = doc_stride,
        max_query_length = max_query_len,
        is_training = is_train,
        return_dataset = "pt")

    if is_train:
      return _get_balanced_dataset(dataset)
    else:
      return dataset


def _get_balanced_dataset(dataset):
    """
    returns a new dataset, where positive and negative examples are approximately balanced
    """
    pos_mask = _get_dataset_pos_mask(dataset)
    neg_mask = [~mask for mask in pos_mask]
    npos, nneg = np.sum(pos_mask), np.sum(neg_mask)

    neg_keep_frac = npos / nneg  # So that in expectation there will be npos negative examples (--> balanced)
    neg_keep_mask = [mask and np.random.random() < neg_keep_frac for mask in neg_mask]

    # keep all positive examples and subset of negative examples
    keep_mask = [pos_mask[i] or neg_keep_mask[i] for i in range(len(pos_mask))]
    keep_indices = [i for i in range(len(keep_mask)) if keep_mask[i]]

    subset_dataset = torch.utils.data.Subset(dataset, keep_indices)
    return subset_dataset


def _get_dataset_pos_mask(dataset):
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


def get_cuad_dataloaders(tokenizer_or_name = "roberta-base",
                         train_batch_size = 8,
                         valid_batch_size = 8,
                         seed = None,
                         **kwargs):
  torch.manual_seed(1234 if seed is None else seed)
  train_dataset = CuadDataset(tokenizer_or_name, is_train=True, **kwargs)
  valid_dataset = CuadDataset(tokenizer_or_name, is_train=False, **kwargs)
  train_loader = tud.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
  valid_loader = tud.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True)
  return { "train_dataset" : trains,
           "valid_dataset" : valids,
           "train_dataloader" : train_loader,
           "valid_dataloader" : valid_loader
          }


