import os
import pathlib

import torch
import torch.utils.data as tud

PROJ_ROOT = pathlib.Path(__file__).parent.parent.resolve().as_posix()
DATA_DIR = os.path.join(PROJ_ROOT, "data")

# Shuffle stuff

def shuffle_dataset(dataset, seed=None):
  assert isinstance(dataset, tud.Dataset)
  _ = torch.manual_seed(seed) if isinstance(seed, int) else None
  perm = torch.randperm(len(dataset))
  return tud.Subset(dataset, indices=perm)


# Assume the class is 0/1, and we want to have balanced 0/1s
def get_tabular_balanced_subset(dataset, seed=None):
  if seed is not None: torch.manual_seed(seed)
  pos_mask = dataset.y  # Only for the tabular data (HAI, WADI, SWaT) do we have y
  neg_mask = 1 - pos_mask
  npos = pos_mask.sum()
  nneg = pos_mask.numel() - npos

  neg_keep_frac = npos / nneg
  neg_keep_mask = neg_mask * (torch.rand(pos_mask.shape) < neg_keep_frac)
  keep_mask = (pos_mask + neg_keep_mask).clamp(0,1)
  keep_inds = keep_mask.nonzero()
  subset = tud.Subset(dataset, indices=keep_inds)
  return subset



