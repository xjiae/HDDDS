import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import os

class WADIDataset(torch.utils.data.Dataset):
    def __init__(self, root="data", contents=None, raw=False):
        assert contents in ["all", "train", "valid"]
        train_fn, test_fn = "wadi/train_processed.csv", "wadi/test_processed.csv"
        test_gt = 'wadi/test_gt_exp.csv'
        raw_fn = 'wadi/raw.csv'
        train_path, test_path = os.path.join(root, train_fn), os.path.join(root, test_fn)
        test_exp_path = os.path.join(root, test_gt)
        raw_path =  os.path.join(root, raw_fn)
        if contents == "all":
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            self.data = pd.concat([train_data, test_data])
            train_explanation = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-3)))
            test_explanation = pd.read_csv(test_exp_path)
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        elif contents == "train":
            self.data = pd.read_csv(train_path)
            self.explanation = pd.DataFrame(np.zeros((self.data.shape[0], self.data.shape[1]-3)))
        elif contents == "valid":
            self.data = pd.read_csv(test_path)
            self.explanation = pd.read_csv(test_exp_path)
        else:
            raise NotImplementedError()

        if raw:
            self.data = pd.read_csv(raw_path)   
            test_explanation = pd.read_csv(test_exp_path)
            train_explanation = pd.DataFrame(np.zeros((self.data.shape[0]-len(test_explanation), test_explanation.shape[1])))
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        self.timestamp = self.data['epoch']
        self.y = torch.tensor(self.data['label'].values)
     
        

    def __getitem__(self, index):
        index = index.item() if isinstance(index, torch.Tensor) else index
        x = torch.tensor(self.data.iloc[index, 1:-2].values)
        y = torch.tensor(self.data.iloc[index, -2])
        w = torch.tensor(self.explanation.iloc[index, :].values)
        return x, y.long(), w
            
       

    def __len__(self):
        return self.data.shape[0]
    
    
class WADISlidingDataset(torch.utils.data.Dataset):
    def __init__(self, window_size, stride=1, contents = None, root = "data"):
        assert contents in ["all", "train", "valid"]
        train_fn, test_fn = "wadi/train_processed.csv", "wadi/test_processed.csv"
        test_gt = 'wadi/test_gt_exp.csv'
        train_path, test_path = os.path.join(root, train_fn), os.path.join(root, test_fn)
        test_exp_path = os.path.join(root, test_gt)
        if contents == "train":
            df =  pd.read_csv(train_path, index_col=0)
            self.explanation = pd.DataFrame(np.zeros((df.shape[0], df.shape[1]-2)))
        elif contents == "valid":
            df = pd.read_csv(test_path, index_col=0)
            self.explanation = pd.read_csv(test_exp_path)
        elif contents == "all":
            train_data = pd.read_csv(train_path, index_col=0)
            test_data = pd.read_csv(test_path, index_col=0)
            df = pd.concat([train_data, test_data])
            train_explanation = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_explanation = pd.read_csv(test_exp_path)
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        else:
            raise NotImplementedError()
        
      
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
                self.y.append(self.labels.values[R])
        self.y = torch.tensor(self.y)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")
        
        self.num_features = df.shape[1] - 2
    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + self.window_size - 1
        WINDOW_GIVEN = self.window_size - 1
        item = {}
        item['y'] = self.y[idx]
        item["ts"] = self.ts[i + self.window_size - 1]
        item["x"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
        item["xl"] = torch.from_numpy(self.tag_values[last])
        # item['exp'] = torch.from_numpy(self.explanation.iloc[idx].values)
        item['exp'] = torch.from_numpy(self.explanation.iloc[i : i + WINDOW_GIVEN].values)
        return item["x"], item['y'], item['exp'], item["xl"]
    
    def get_ts(self):
        return self.ts    
    


def get_wadi_dataloaders(normalize = False,
                        train_batch_size = 32,
                        valid_batch_size = 32,
                        mix_good_and_anom = True,
                        train_frac = 0.7,
                        shuffle = True,
                        seed = None):
  good_dataset = WADIDataset(contents="train", raw=normalize)
  anom_dataset = WADIDataset(contents="valid", raw=normalize)

  torch.manual_seed(1234 if seed is None else seed)
  if mix_good_and_anom:
    concats = tud.ConcatDataset([good_dataset, anom_dataset])
    total = len(concats)
    num_train = int(total * train_frac)
    trains, valids = tud.random_split(concats, [num_train, total - num_train])
  else:
    trains, valids = good_dataset, anom_dataset

  trains_perm = torch.randperm(len(trains)) if shuffle else torch.tensor(range(len(trains)))
  valids_perm = torch.randperm(len(valids)) if shuffle else torch.tensor(range(len(valids)))
  trains = tud.Subset(trains, indices=trains_perm)
  valids = tud.Subset(valids, indices=valids_perm)

  train_loader = tud.DataLoader(trains, batch_size=train_batch_size, shuffle=shuffle)
  valid_loader = tud.DataLoader(valids, batch_size=valid_batch_size, shuffle=shuffle)
  return { "train_dataset" : trains,
           "valid_dataset" : valids,
           "train_dataloader" : train_loader,
           "valid_dataloader" : valid_loader }


def get_wadi_sliding_dataloaders(window_size,
                                 stride = 1,
                                 train_batch_size = 32,
                                 valid_batch_size = 32,
                                 mix_good_and_anom = True,
                                 train_frac = 0.7,
                                 seed = None):
  good_dataset = WADISlidingDataset(window_size=window_size, stride=stride, contents="train")
  anom_dataset = WADISlidingDataset(window_size=window_size, stride=stride, contents="valid")

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
           "valid_dataloader" : valid_loader
          }



