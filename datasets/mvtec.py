import os
from glob import glob

import torch
import torch.utils.data as tud
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

class MVTecItemDataset(Dataset):
    def __init__(self,
                 category,
                 root = os.path.join(DATA_DIR, "mvtec-ad"),
                 input_size = 256,  # Loads as (3,256,256) images
                 contents = "train",
                 good_value = 0,
                 anom_value = 1):

        if "metal" in category:
            category = "metal_nut"
        assert category in MVTEC_CATEGORIES
        self.image_transform = transforms.Compose(
            [ transforms.Resize(input_size),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        self.target_transform = transforms.Compose(
            [ transforms.Resize(input_size),
              transforms.ToTensor(),
            ])

        assert contents in ["train", "test"]
        self.contents = contents
        if contents == "train":
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))

        self.good_value = good_value
        self.anom_value = anom_value


    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert("RGB") if image.mode == "L" else image
        image = self.image_transform(image)
        if self.contents == "train":
            target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            return image, self.good_value, target
        else:
            y = self.good_value
            if os.path.dirname(image_file).endswith("good"):
              target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
              target = Image.open(
                  image_file.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")
              )
              target = self.target_transform(target)
              y = self.anom_value

            return image, y, target

    def __len__(self):
        return len(self.image_files)


# Returns the train and validation dataset
def get_mvtec_bundle(categories=["all"],
                     train_batch_size = 8,
                     test_batch_size = 8,
                     train_has_only_goods = False,
                     train_frac = 0.7,
                     shuffle = True,
                     seed = 1234):
    good_datasets = []
    anom_datasets = []
    if "all" in categories:
        categories = MVTEC_CATEGORIES
    for cat in categories:
        assert cat in MVTEC_CATEGORIES
        good_datasets.append(MVTecItemDataset(cat, contents="train"))
        anom_datasets.append(MVTecItemDataset(cat, contents="test"))

    torch.manual_seed(seed)
    if train_has_only_goods:
        trains = tud.ConcatDataset(good_dataset)
        tests = tud.ConcatDataset(anom_dataset)
    else:
        concats = tud.ConcatDataset(good_datasets + anom_datasets)
        total = len(concats)
        num_train = int(total * train_frac)
        trains, tests = tud.random_split(concats, [num_train, total - num_train])

    trains_perm = torch.randperm(len(trains)) if shuffle else torch.tensor(range(len(trains)))
    tests_perm = torch.randperm(len(tests)) if shuffle else torch.tensor(range(len(tests)))
    trains = tud.Subset(trains, indices=trains_perm)
    tests = tud.Subset(tests, indices=tests_perm)

    train_dataloader = tud.DataLoader(trains, batch_size=train_batch_size, shuffle=shuffle)
    test_dataloader = tud.DataLoader(tests, batch_size=test_batch_size, shuffle=shuffle)
    return { "train_dataset" : trains,
             "test_dataset" : tests,
             "train_dataloader" : train_dataloader,
             "test_dataloader" : test_dataloader
           }


