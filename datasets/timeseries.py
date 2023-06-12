import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import os

from datetime import datetime

from .dataset_utils import *

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, ds_name, window_size,
                 stride = 1,
                 root = "data",
                 contents = "all",
                 label_choice = "all",
                 overwrite_cache = False):
        print(f"Entering constructor call at {datetime.now()}")
        ds_name = ds_name.lower()
        assert ds_name in ["hai", "swat", "wadi"]
        assert contents in ["all", "train", "test", "raw"]
        assert label_choice in ["all", "last", "exist"]

        self.data_dir = os.path.join(root, ds_name)
        self.train_file = os.path.join(self.data_dir, "train_processed.csv")
        self.test_file = os.path.join(self.data_dir, "test_processed.csv")
        self.test_explanation_file = os.path.join(self.data_dir, "test_gt_exp.csv")
        self.raw_file = os.path.join(self.data_dir, "raw.csv")
        self.contents = contents
        self.window_size = window_size
        self.stride = stride
        self.label_choice = label_choice

        # Set up cache if not exists
        self.cache_dir = os.path.join(self.data_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Get cache file if exists
        self.cache_file = f"{ds_name}_W{window_size}_C{contents}_L{label_choice}.cache"
        if os.path.exists(self.cache_file) and not overwrite_cache:
            print(f"loading cache from {self.cache_file}")
            processed_dict = torch.load(self.cache_file)
        else:
            processed_dict = self.make_processed_dict()

        self.num_features = processed_dict["num_features"]
        self.time_points = processed_dict["time_points"]      # (T,)
        self.features = processed_dict["features"]            # (T, num_features)
        self.window_start_indices = processed_dict["window_start_indices"] # (num_windows,)
        self.window_labels = processed_dict["window_labels"]  # (num_windows,) or (num_windows,window_size)
        self.num_windows = self.windows.size(0)


    def make_processed_dict(self):
        print(f"About to load df {datetime.now()}")
        if self.contents == "train":
            df =  pd.read_csv(self.train_file, index_col=0)
            explanation = pd.DataFrame(np.zeros((df.shape[0], df.shape[1]-2))) 
        elif self.contents == "test":
            df = pd.read_csv(self.test_file, index_col=0)
            explanation = pd.read_csv(self.test_explanation_file)
        elif self.contents == "all":
            train_data = pd.read_csv(self.train_file, index_col=0)
            test_data = pd.read_csv(self.test_file, index_col=0)
            df = pd.concat([train_data, test_data])
            train_exp = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_exp = pd.read_csv(self.test_explanation_file)
            explanation = pd.DataFrame(np.vstack([train_exp.values, test_exp.values]), columns=test_exp.columns)
        elif self.contents == "raw":
            df =  pd.read_csv(self.raw_file, index_col=0)
            train_data = pd.read_csv(self.train_file, index_col=0)
            train_exp = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_exp = pd.read_csv(self.test_explanation_file)
            explanation = pd.DataFrame(np.vstack([train_exp.values, test_exp.values]), columns=test_exp.columns)
        else:
            raise NotImplementedError()

        print(f"Donw loading dict at {datetime.now()}")
        num_features = df.shape[1] - 2
        time_points = torch.from_numpy(df["epoch"].values).long()
        print(f"Finished time points at {datetime.now()}")
        labels = torch.from_numpy(df["label"].values).long()
        print(f"Finished labels at {datetime.now()}")
        features = torch.from_numpy(df.drop(columns=["epoch", "label"]).values).float()

        window_start_indices = []
        window_labels = []
        print(f"About to run sally's for loop at {datetime.now()}")
        for L in range(len(time_points) - self.window_size + 1):
            R = L + self.window_size - 1
            if (time_points[R] - time_points[L]) == self.window_size - 1:
                window_start_indices.append(L)
                if self.label_choice == "last":
                    window_labels.append(labels[R])
                elif self.label_choice == "all":
                    window_labels.append(labels[L:R+1])
                elif self.label_choice == "exist":
                    window_labels.append(max(labels[L:R+1]))
                else:
                    raise NotImplementedError()
        print(f"Done running for loop at {datetime.now()}")

        window_start_indices = torch.from_numpy(np.array(window_start_indices)[::self.stride]).long()
        window_labels = torch.from_numpy(np.array(window_labels, dtype=np.int32)[::self.stride]).long()

        if self.label_choice == "all":
            window_labels = window_labels.view(-1, self.window_size)
        else:
            window_labels = window_labels.view(-1)

        return {
            "num_features" : num_features,
            "time_points" : time_points,
            "features" : features,
            "window_start_indices" : window_start_indices,
            "window_labels" : window_labels
        }



        '''
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
        '''
         
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


def process_dataframe_windows(contents_to_files, window_size, contents, label_choice):
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


