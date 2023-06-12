import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import os

from .dataset_utils import *

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, ds_name, window_size,
                 stride = 1,
                 root = "data",
                 contents = "all",
                 label_choice = "all",
                 overwrite_cache = False):
        ds_name = ds_name.lower()
        assert ds_name in ["hai", "swat", "wadi"]
        assert contents in ["all", "train", "test", "raw"]
        assert label_choice in ["all", "last", "exist"]
        train_fn, test_fn = f"{ds_name}/train_processed.csv", f"{ds_name}/test_processed.csv"
        test_gt = f'{ds_name}/test_gt_exp.csv'
        raw_fn = f'{ds_name}/raw.csv'
        train_path, test_path = os.path.join(root, train_fn), os.path.join(root, test_fn)
        test_exp_path = os.path.join(root, test_gt)
        raw_path =  os.path.join(root, raw_fn)
        self.contents = contents
        if contents == "train":
            df =  pd.read_csv(train_path, index_col=0)
            self.explanation = pd.DataFrame(np.zeros((df.shape[0], df.shape[1]-2))) 
        elif contents == "test":
            df = pd.read_csv(test_path, index_col=0)
            self.explanation = pd.read_csv(test_exp_path)
        elif contents == "all":
            train_data = pd.read_csv(train_path, index_col=0)
            test_data = pd.read_csv(test_path, index_col=0)
            df = pd.concat([train_data, test_data])
            train_exp = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_exp = pd.read_csv(test_exp_path)
            self.explanation = pd.DataFrame(np.vstack([train_exp.values, test_exp.values]), columns=test_exp.columns)
        elif contents == "raw":
            df =  pd.read_csv(raw_path, index_col=0)
            train_data = pd.read_csv(train_path, index_col=0)
            train_exp = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_exp = pd.read_csv(test_exp_path)
            self.explanation = pd.DataFrame(np.vstack([train_exp.values, test_exp.values]), columns=test_exp.columns)
        else:
            raise NotImplementedError()
        self.num_features = df.shape[1] - 2
        self.ts = df['epoch'].values
        self.labels = df['label']
        self.window_size = window_size
        df_tag = df.drop(columns=['epoch', 'label'])
        self.tag_values = np.array(df_tag, dtype=np.float32)
        self.test_idxs = []
        self.y = []
        
        for L in range(len(self.ts) - self.window_size + 1):
            R = L + self.window_size - 1
            if (self.ts[R]-self.ts[L]) == self.window_size - 1:
                self.test_idxs.append(L)
                if label_choice == "last":
                    self.y.append(self.labels.values[R])
                elif label_choice == "all":
                    self.y.append(self.labels.values[L:R+1])
                elif label_choice == "exist":
                    self.y.append(max(self.labels.values[L:R+1]))
                else:
                    raise NotImplementedError()

        self.y = torch.tensor(np.array(self.y)).long()
        if label_choice == "all":
          self.y = self.y.view(-1, window_size)
        else:
          self.y = self.y.view(-1)


        self.test_idxs = np.array(self.test_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.test_idxs)
        print(f"# of {self.contents} windows: {self.n_idxs}")
         
    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.test_idxs[idx]
        y = self.y[idx].view(-1)
        x = torch.from_numpy(self.tag_values[i : i + self.window_size])
        w = torch.from_numpy(self.explanation.iloc[i : i + self.window_size].values)
        return x, y, w
    
    def get_ts(self):
        return self.ts


def get_timeseries_bundle(ds_name,
                          window_size=100,
                          label_choice = "all",
                          stride = 1,
                          train_batch_size = 32,
                          test_batch_size = 32,
                          train_has_only_goods = False,
                          train_frac = 0.7,
                          shuffle = True,
                          seed = 1234):

    good_dataset = TimeSeriesDataset(ds_name,
                                     window_size,
                                     stride = stride,
                                     contents = "train",
                                     label_choice = label_choice)
    anom_dataset = TimeSeriesDataset(ds_name,
                                     window_size,
                                     stride = stride,
                                     contents = "test",
                                     label_choice = label_choice)

    torch.manual_seed(seed)
    if train_has_only_goods:
        trains, tests = good_dataset, anom_dataset
    else:
        concats = tud.ConcatDataset([good_dataset, anom_dataset])
        good_y = good_dataset.y
        anom_y = anom_dataset.y
        # In this case, label is (N,window_size)
        if label_choice == "all":
          good_y = good_y.max(dim=1).values
          anom_y = anom_y.max(dim=1).values

        pos_mask = torch.cat([good_y, anom_y])
        balanced = get_binary_balanced_subset(concats, pos_mask)

        total = len(balanced)
        num_train = int(total * train_frac)
        trains, tests = tud.random_split(balanced, [num_train, total - num_train])

    train_dataloader = tud.DataLoader(trains, batch_size=train_batch_size, shuffle=shuffle)
    test_dataloader = tud.DataLoader(tests, batch_size=test_batch_size, shuffle=shuffle)
    return { "train_dataset" : trains,
             "test_dataset" : tests,
             "train_dataloader" : train_dataloader,
             "test_dataloader" : test_dataloader
           }


