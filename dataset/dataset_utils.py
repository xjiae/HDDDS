import os
import pathlib

import torch
import torch.utils.data as tudata

PROJ_ROOT = pathlib.Path(__file__).parent.parent.resolve().as_posix()
DATA_DIR = os.path.join(PROJ_ROOT, "data")

# Shuffle stuff

def shuffle_dataset(dataset, seed=None):
  assert isinstance(dataset, tudata.Dataset)
  _ = torch.manual_seed(seed) if isinstance(seed, int) else None
  perm = torch.randperm(len(dataset))
  return tudata.Subset(dataset, indices=perm)


