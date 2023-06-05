import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.utils.data as tud
import os

WINDOW_SIZE = 40
WINDOW_GIVEN = 39


class HAIDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, contents=None, raw=False):
        assert contents in ["all", "train", "valid"]
        if contents == "all":
            train_data = pd.read_csv('data/hai/train_processed.csv')
            test_data = pd.read_csv('data/hai/test_processed.csv')
            self.data = pd.concat([train_data, test_data])
            train_explanation = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-3)))
            test_explanation = pd.read_csv('data/hai/test_gt_exp.csv')
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        elif contents == "train":
            self.data = pd.read_csv('data/hai/train_processed.csv')
            self.explanation = pd.DataFrame(np.zeros((self.data.shape[0], self.data.shape[1]-3)))
        elif contents == "valid":
            self.data = pd.read_csv('data/hai/test_processed.csv')
            self.explanation = pd.read_csv('data/hai/test_gt_exp.csv')
        else:
            raise NotImplementedError()

        if raw:
            self.data = pd.read_csv('data/hai/raw.csv')   
            test_explanation = pd.read_csv('data/hai/test_gt_exp.csv')
            train_explanation = pd.DataFrame(np.zeros((self.data.shape[0]-len(test_explanation), test_explanation.shape[1])))
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        self.timestamp = self.data['epoch']
        self.y = self.data['label']
     
        

    def __getitem__(self, index):
        return self.data.iloc[index, 1:-2].values, self.data.iloc[index, -2], self.explanation.iloc[index, :].values
            
       

    def __len__(self):
        return self.data.shape[0]
    
    def get_ts(self):
        return self.timestamp


class HAISlidingDataset(torch.utils.data.Dataset):
    def __init__(self, window_size, stride=1, contents=None):
        assert contents in ["all", "train", "valid"]
        if contents == "train":
            df =  pd.read_csv('data/hai/train_processed.csv', index_col=0)
            self.explanation = pd.DataFrame(np.zeros((df.shape[0], df.shape[1]-2)))
            
        elif contents == "valid":
            df = pd.read_csv('data/hai/test_processed.csv', index_col=0)
            self.explanation = pd.read_csv('data/hai/test_gt_exp.csv')
        elif contents == "all":
            train_data = pd.read_csv('data/hai/train_processed.csv')
            test_data = pd.read_csv('data/hai/test_processed.csv')
            df = pd.concat([train_data, test_data])
            train_explanation = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-3)))
            test_explanation = pd.read_csv('data/hai/test_gt_exp.csv')
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
                if 1 in self.labels[L : L + self.window_size - 1].values:
                    self.y.append(1)
                else:
                    self.y.append(0)
                
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
        # idx = last
        # if 1 in self.labels[i : i + WINDOW_GIVEN].values:
        #     item['y'] = 1
        #     idx = np.where(self.labels[i : i + WINDOW_GIVEN] == 1)[0][0] + last
        item["ts"] = self.ts[i + self.window_size - 1]
        item["x"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
        item["xl"] = torch.from_numpy(self.tag_values[last])
        # item['exp'] = torch.from_numpy(self.explanation.iloc[idx].values)
        item['exp'] = torch.from_numpy(self.explanation.iloc[i : i + WINDOW_GIVEN].values)
        
        return item["x"], item['y'], item['exp'], item["xl"]
    
    def get_ts(self):
        return self.ts

# max min(0-1)
def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    return train_ret, test_ret


# downsample by 10
def downsample(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()
    # print('before downsample', np_data.shape)

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))

    d_data = d_data.transpose()

    # print('after downsample', d_data.shape, d_labels.shape)

    return d_data.tolist(), d_labels.tolist()


def process_gt():
    df = pd.read_csv('../data/hai/hai_gt.csv')
    # breakpoint()
    # f = lambda x: len(x.iloc[:,0].split(" "))-1
    # df['date'] = df.iloc[:,0].str.split(" ").str[0]
    # df['end_date'] = df.date.astype(str)+' '+ df.iloc[:,1]
    
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.to_datetime(df["time"]).dt.time
    combined_datetime = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    second = pd.to_timedelta(df['Duration (sec)'], unit='s')
    df['start_epoch'] = combined_datetime.astype('int64')//1e9
    # df['start_epoch'] = pd.to_datetime(df['start']).astype('int64')//1e9   
    df['end_epoch'] = pd.to_datetime(combined_datetime + second).astype('int64')//1e9 
    return df

def look_up(gt, data, time):
    
    for j in range(len(gt)):
        if gt.end_epoch.iloc[j] < time:
            continue
        else:
            if gt.start_epoch.iloc[j] <= time:
                attacked = gt['target'].iloc[j]
                if ',' in attacked:
                    attacked = attacked.split(", ")
                else:
                    attacked = [attacked]
                attacked_index = np.where(data.columns.isin(attacked))[0]
                
                # data.columns.index(data.columns.isin(attacked))
                return attacked_index
    return  -1
    
def preprocess(path, train=True):
    files = os.listdir(path)
    dfs = []
    for f in tqdm(files): 
        if 'summary' in f:
            continue
        if train and 'test' in f:
            continue
        if train == False and 'train' in f:
            continue
        # breakpoint()
        df  = pd.read_csv(path+f)
        df["epoch"] = pd.to_datetime(df["timestamp"]).astype('int64')//1e9
        df = df.drop(columns=['timestamp'])
        df = df.fillna(df.mean())
        df = df.fillna(0)
        df = df.rename(columns=lambda x: x.strip())
        dfs.append(df)
    df = pd.concat(dfs)
    return df

def normalize(df, min, max):
        ndf = df.copy()
        for c in df.columns:
            if min[c] == max[c]:
                ndf[c] = df[c] - min[c]
            else:
                ndf[c] = (df[c] - min[c]) / (max[c] - min[c])
        return ndf


def get_hai_dataloaders(normalize = False,
                        train_batch_size = 32,
                        valid_batch_size = 32,
                        test_batch_size = 32,
                        mix_good_and_anom = True,
                        train_frac = 0.7,
                        test_frac = 0.2,
                        seed = None):
  good_dataset = HAIDataset(contents="train", raw=normalize)
  anom_dataset = HAIDataset(contents="valid", raw=normalize)

  torch.manual_seed(1234 if seed is None else seed)
  if mix_good_and_anom:
    concats = tud.ConcatDataset([good_dataset, anom_dataset])
    total = len(concats)
    num_train = int(total * train_frac)
    num_test = int(total * test_frac)
    trains, valids = tud.random_split(concats, [num_train, total - num_train])
    # valids, tests = tud.random_split(valids, [total - num_train - num_test, num_test])
  else:
    trains, valids = good_dataset, anom_dataset
    # total = len(valids)
    # valids, tests = tud.random_split(valids, [total - num_test, num_test])
    

  train_loader = tud.DataLoader(trains, batch_size=train_batch_size, shuffle=True)
  valid_loader = tud.DataLoader(valids, batch_size=valid_batch_size, shuffle=True)
#   test_loader = tud.DataLoader(tests, batch_size=valid_batch_size, shuffle=True)
  return { "train_dataset" : trains,
           "valid_dataset" : valids,
           "test_dataset" : tests,
           "train_dataloader" : train_loader,
           "valid_dataloader" : valid_loader,
        #    "test_dataloader" : test_loader
          }


def get_hai_sliding_dataloaders(window_size,
                                stride = 1,
                                train_batch_size = 32,
                                valid_batch_size = 32,
                                mix_good_and_anom = True,
                                train_frac = 0.7,
                                seed = None):
  good_dataset = HAISlidingDataset(window_size=window_size, stride=stride, contents="train")
  anom_dataset = HAISlidingDataset(window_size=window_size, stride=stride, contents="valid")

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



def main():
    gt = process_gt()
    path = '../../../../data2/xjiae/hddds/hai/hai-22.04/'
    train = preprocess(path)
    test = preprocess(path, False)

    
    test_labels = test['Attack'].values
    
    # train_mask = []
    # for i in tqdm(range(len(train))):
    #     mask = np.zeros(train.shape[1]-2)
    #     train_mask.append(mask)
  
    test_mask = []
    for i in tqdm(range(len(test))):
        if test_labels[i] == 0:
            mask = np.zeros(test.shape[1]-2)
        else:
            attacked = look_up(gt, test, test.epoch.iloc[i])
            mask = np.zeros(test.shape[1]-2)
            if type(attacked) != int:
                mask[attacked] = 1
            if mask.sum() == 86:
                breakpoint()
            
        test_mask.append(mask)
    
    # test = test.drop(columns = ['epoch'])
    # train = train.drop(columns = ['epoch'])
    
    
    # combined = pd.concat([train, test])
    # combined = combined.rename(columns={'Attack':'label'})
    # combined.to_csv("../data/hai/raw.csv")
    
    # # breakpoint()
    
    # train = train.rename(columns={'Attack':'label'})
    # test = test.rename(columns={'Attack':'label'})
    
    
    # train_df = train.drop(columns=['epoch', 'label'])
    # test_df = test.drop(columns=['label', 'epoch'])
    
    
    # min = train_df.min()
    # max = train_df.max()    
        
    # train_df = normalize(train_df, min, max).ewm(alpha=0.9).mean()
    # test_df = normalize(test_df, min, max).ewm(alpha=0.9).mean()
    
    # train_df['label'] = train['label']
    # train_df['epoch'] = train['epoch']
    
    
    # test_df['label'] = test['label']
    # test_df['epoch'] = test['epoch']
        
    # train_df.to_csv("data/hai/train_processed.csv")
    # test_df.to_csv("data/hai/test_processed.csv")
    
    df = pd.DataFrame(test_mask)
    df.to_csv('../data/hai/test_gt_exp.csv', index = False)
    breakpoint()
    
    
    
    
    
    # df = pd.DataFrame(all_mask)
    # df.to_csv('swat_gt_exp.csv', index = False)
    
        
    
    
        

    # train = train.iloc[:, 2:]
    # test = test.iloc[:, 3:]


    # train = train.fillna(train.mean())
    # test = test.fillna(test.mean())
    # train = train.fillna(0)
    # test = test.fillna(0)

    # # trim column names
    # train = train.rename(columns=lambda x: x.strip())
    # test = test.rename(columns=lambda x: x.strip())

    # breakpoint()
    # train_labels = np.zeros(len(train))
    # test_labels = test.attack

    # # train = train.drop(columns=['attack'])
    # test = test.drop(columns=['attack'])




    # x_train, x_test = norm(train.values, test.values)


    # for i, col in enumerate(train.columns):
    #     train.loc[:, col] = x_train[:, i]
    #     test.loc[:, col] = x_test[:, i]



    # d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    # d_test_x, d_test_labels = downsample(test.values, test_labels, 10)

    # train_df = pd.DataFrame(d_train_x, columns = train.columns)
    # test_df = pd.DataFrame(d_test_x, columns = test.columns)


    # test_df['attack'] = d_test_labels
    # train_df['attack'] = d_train_labels

    # train_df = train_df.iloc[2160:]

    # train_df.to_csv('./train.csv')
    # test_df.to_csv('./test.csv')

    # f = open('./list.txt', 'w')
    # for col in train.columns:
    #     f.write(col+'\n')
    # f.close()

if __name__ == '__main__':
    main()
