import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import os

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, ds_name, window_size, stride=1, root="data", contents=None, label_choice=None):
        assert ds_name in ["hai", "swat", "wadi"]
        assert contents in ["all", "train", "test", "raw"]
        assert label_choice in ["all", "last", "exist"]
        train_fn, test_fn = f"{ds_name}/train_processed.csv", f"{ds_name}/test_processed.csv"
        test_gt = f'{ds_name}/test_gt_exp.csv'
        raw_fn = f'{ds_name}/raw.csv'
        train_path, test_path = os.path.join(root, train_fn), os.path.join(root, test_fn)
        test_exp_path = os.path.join(root, test_gt)
        raw_path =  os.path.join(root, raw_fn)
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
            train_explanation = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_explanation = pd.read_csv(test_exp_path)
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        elif contents == "raw":
            df =  pd.read_csv(raw_path, index_col=0)
            train_data = pd.read_csv(train_path, index_col=0)
            train_explanation = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_explanation = pd.read_csv(test_exp_path)
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        else:
            raise NotImplementedError()
        self.num_features = df.shape[1] - 2
        self.ts = df['epoch'].values
        self.labels = df['label']
        self.window_size = window_size
        df_tag = df.drop(columns=['epoch', 'label'])
        self.tag_values = np.array(df_tag, dtype=np.float32)
        self.valid_idxs = []
        self.y = []
        
        for L in range(len(self.ts) - self.window_size + 1):
            R = L + self.window_size - 1
            if (self.ts[R]-self.ts[L]) == self.window_size - 1:
                self.valid_idxs.append(L)
                if label_choice == "last":
                    self.y.append(self.labels.values[R])
                elif label_choice == "all":
                    self.y.append(self.labels.values[L:R+1])
                elif label_choice == "exist":
                    self.y.append(max(self.labels.values[L:R+1]))
                else:
                    raise NotImplementedError()
        self.y = torch.tensor(self.y).long()
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")
         
    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        item = {}
        item['y'] = self.y[idx]
        item["ts"] = self.ts[i + self.window_size - 1]
        item["x"] = torch.from_numpy(self.tag_values[i : i + self.window_size])
        item['w'] = torch.from_numpy(self.explanation.iloc[i : i + self.window_size].values)
        return item["x"], item['y'], item['w']
    
    def get_ts(self):
        return self.ts


def get_timeseries_dataloaders(window_size,
                                stride = 1,
                                train_batch_size = 32,
                                valid_batch_size = 32,
                                mix_good_and_anom = True,
                                train_frac = 0.7,
                                seed = None):
  good_dataset = TimeSeriesDataset(window_size=window_size, stride=stride, contents="train")
  anom_dataset = TimeSeriesDataset(window_size=window_size, stride=stride, contents="test")

  torch.manual_seed(1234 if seed is None else seed)
  if mix_good_and_anom:
    concats = tud.ConcatDataset([good_dataset, anom_dataset])
    total = len(concats)
    num_train = int(total * train_frac)
    trains, valids = tud.random_split(concats, [num_train, total - num_train])
  else:
    trains, valids = good_dataset, anom_dataset

  train_loader = tud.DataLoader(trains, batch_size=train_batch_size, shuffle=False)
  valid_loader = tud.DataLoader(valids, batch_size=valid_batch_size, shuffle=False)
  return { "train_dataset" : trains,
           "valid_dataset" : valids,
           "train_dataloader" : train_loader,
           "valid_dataloader" : valid_loader }

