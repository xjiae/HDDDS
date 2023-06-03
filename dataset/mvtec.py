import os
from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .dataset_utils import *

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

class MVTecDataset(Dataset):
    def __init__(self,
                 category,
                 root = os.path.join(DATA_DIR, "mvtec-ad"),
                 input_size = 256,  # Loads as (3,256,256) images
                 is_train = True):
        assert category in MVTEC_CATEGORIES
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train
        self.good_value = -1
        self.anom_value = 1

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        if image.mode == "L":
            image = image.convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            return image, self.good_value, target
        else:
            y = self.good_value
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                )
                target = self.target_transform(target)
                y = self.anom_value

            return image, y, target

    def __len__(self):
        return len(self.image_files)


# Returns the train and validation dataset
def get_mvtec_dataloaders(categories,
                          train_batch_size = 8,
                          valid_batch_size = 8,
                          train_frac = 0.7,
                          mix_good_and_anom = True,
                          seed = None):
    good_datasets = []
    anom_datasets = []
    if "all" in categories:
        categories = MVTEC_CATEGORIES
    for cat in categories:
        assert cat in MVTEC_CATEGORIES
        good_datasets.append(MVTecDataset(cat, is_train=True))
        anom_datasets.append(MVTecDataset(cat, is_train=False))

    torch.manual_seed(1234 if seed is None else seed)
    if mix_good_and_anom:
      concats = torch.utils.data.ConcatDataset(good_datasets + anom_datasets)
      total = len(concats)
      num_train = int(total * train_frac)
      trains, valids = torch.utils.data.random_split(concats, [num_train, total - num_train])
    else:
      trains = torch.utils.data.ConcatDataset(good_dataset)
      valids = torch.utils.data.ConcatDataset(anom_dataset)

    train_loader = torch.utils.data.DataLoader(trains, batch_size=train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valids, batch_size=valid_batch_size, shuffle=True)
    return trains, valids, train_loader, valid_loader


