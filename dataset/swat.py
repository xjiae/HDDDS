import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import torch
import torch.utils.data as tud
from tqdm import tqdm

class SWaTDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, contents=None, raw = False):
        assert contents in ["all", "train", "valid"]
        if contents == "all":
            train_data = pd.read_csv('data/swat/train_processed.csv', index_col=0)
            test_data = pd.read_csv('data/swat/test_processed.csv', index_col=0)
            self.data = pd.concat([train_data, test_data])
            train_explanation = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-2)))
            test_explanation = pd.read_csv('data/swat/test_gt_exp.csv')
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        elif contents == "train":
            self.data = pd.read_csv('data/swat/train_processed.csv', index_col=0)
            self.explanation = pd.DataFrame(np.zeros((self.data.shape[0], self.data.shape[1]-2)))
        elif contents == "valid":
            self.data = pd.read_csv('data/swat/test_processed.csv', index_col=0)
            self.explanation = pd.read_csv('data/swat/test_gt_exp.csv')
        else:
            raise NotImplementedError()


        if raw:
            self.data = pd.read_csv('data/swat/raw.csv', index_col=0)   
            test_explanation = pd.read_csv('data/swat/test_gt_exp.csv')
            train_explanation = pd.DataFrame(np.zeros((self.data.shape[0]-len(test_explanation), test_explanation.shape[1])))
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        self.timestamp = self.data['epoch']
        self.label = self.data['label']
        
    def __getitem__(self, index):
        return self.data.iloc[index, :-2].values, self.data.iloc[index, -2], self.explanation.iloc[index, :].values
            
       

    def __len__(self):
        return self.data.shape[0]
    
    def get_ts(self):
        return self.timestamp
    
class SWaTSlidingDataset(torch.utils.data.Dataset):
    def __init__(self, window_size, stride=1, train = True):
        if train:
            df =  pd.read_csv('data/swat/train_processed.csv', index_col=0)
            self.explanation = pd.DataFrame(np.zeros((df.shape[0], df.shape[1]-2)))
        else:
            df = pd.read_csv('data/swat/test_processed.csv', index_col=0)
            self.explanation = pd.read_csv('data/swat/test_gt_exp.csv')
            attacks = df['label'].values
        
      
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

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))


    d_data = d_data.transpose()

    return d_data.tolist(), d_labels.tolist()

def process_gt():
    df = pd.read_csv('../data/swat/swat_gt.csv')
    # f = lambda x: len(x.iloc[:,0].split(" "))-1
    df['date'] = df.iloc[:,0].str.split(" ").str[0]
    df['end_date'] = df.date.astype(str)+' '+ df.iloc[:,1]
    df['end_epoch'] = pd.to_datetime(df['end_date']).astype('int64')//1e9
    df['start_epoch'] = pd.to_datetime(df['Start Time']).astype('int64')//1e9
    
    return df
    
    
    
def look_up(gt, data, time):

    for j in range(len(gt)):
        if gt.end_epoch.iloc[j] < time:
            continue
        else:
            if gt.start_epoch.iloc[j] <= time:
                attacked = gt['Attack Point'].iloc[j]
                if ',' in attacked:
                    attacked = attacked.split(", ")
                else:
                    attacked = [attacked]
                attacked_index = np.where(data.columns.isin(attacked))[0]
                # breakpoint()
                # data.columns.index(data.columns.isin(attacked))
                return attacked_index
    return -1

def normalize(df, min, max):
        ndf = df.copy()
        for c in df.columns:
            if min[c] == max[c]:
                ndf[c] = df[c] - min[c]
            else:
                ndf[c] = (df[c] - min[c]) / (max[c] - min[c])
        return ndf

def get_swat_dataloaders(normalize = False,
                         train_batch_size = 32,
                         valid_batch_size = 32,
                         mix_good_and_anom = True,
                         train_frac = 0.7,
                         seed = None):
  good_dataset = SWaTDataset(contents="train", raw=normalize)
  anom_dataset = SWaTDataset(contents="valid", raw=normalize)

  torch.manual_seed(1234 if seed is None else seed)
  if mix_good_and_anom:
    concats = tud.ConcatDataset([good_dataset, anom_dataset])
    total = len(concats)
    num_train = int(total * train_frac)
    trains, valids = tud.random_split(concats, [num_train, total - num_train])
  else:
    trains, valids = good_dataset, anom_dataset

  train_loader = tud.DataLoader(trains, batch_size=train_batch_size, shuffle=True)
  valid_loader = tud.DataLoader(valids, batch_size=valid_batch_size, shuffle=True)
  return { "train_dataset" : trains,
           "valid_dataset" : valids,
           "train_dataloader" : train_loader,
           "valid_dataloader" : valid_loader
          }


def get_swat_sliding_dataloaders(window_size,
                                 stride = 1,
                                 train_batch_size = 32,
                                 valid_batch_size = 32,
                                 mix_good_and_anom = True,
                                 train_frac = 0.7,
                                 seed = None):
  good_dataset = SWaTSlidingDataset(window_size=window_size, stride=stride, train=True)
  anom_dataset = SWaTSlidingDataset(window_size=window_size, stride=stride, train=False)

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

    test = pd.read_csv('../../../../data2/xjiae/hddds/swat/swat_test.csv', index_col=0)
    train = pd.read_csv('../../../../data2/xjiae/hddds/swat/swat_train.csv', index_col=0)
    
   


    # test = test.iloc[:, 1:]
    # train = train.iloc[:, 1:]
    
    test['date'] = pd.to_datetime(test.index)
    test['epoch'] = test['date'].astype('int64')//1e9
    
    train['date'] = pd.to_datetime(train.index)
    train['epoch'] = train['date'].astype('int64')//1e9
     
    # breakpoint()

    
    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())
    train_labels = train.attack
    test_labels = test.attack
    train = train.drop(columns=['date'])
    test = test.drop(columns=['date'])

    # print(len(test.columns),test.columns)
    # print(len(train.columns),train.columns)
    
    # for i in range(len(train)):
    #     mask = np.zeros(train.shape[1])
    #     all_mask.append(mask)
    all_mask = []    
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
        all_mask.append(mask)


    
    train_df = train.fillna(train.mean())
    test_df = test.fillna(test.mean())
    
    combined = pd.concat([train_df, test_df])
    combined = combined.rename(columns={'attack':'label'})
    combined.to_csv("../data/swat/raw.csv")
    
    
    train_df = train_df.drop(columns=['attack', 'epoch'])
    test_df = test_df.drop(columns=['attack', 'epoch'])
    
    min = train_df.min()
    max = train_df.max()  
    
    
    train_df = normalize(train_df, min, max).ewm(alpha=0.9).mean()
    test_df = normalize(test_df, min, max).ewm(alpha=0.9).mean()
    
    
    train_df['label'] = train_labels
    train_df['epoch'] = train['epoch']
    
    
    test_df['label'] = test_labels
    test_df['epoch'] = test['epoch']
        
    train_df.to_csv("../data/swat/train_processed.csv")
    test_df.to_csv("../data/swat/test_processed.csv")
    df = pd.DataFrame(all_mask)
    df.to_csv('../data/swat/test_gt_exp.csv', index = False)
    
    
    
    train['label'] = train_labels
    test['label'] = test_labels
    
    
    
    
    
    
    
    
    combined = pd.concat([train, test])
    combined.to_csv("../data/swat/swat_processed.csv")

    
    # train = train.fillna(0)
    # test = test.fillna(0)

            
         
        # for j in range(len(gt)):
        #     if train['epoch'].iloc[i] >= gt['start_epoch'].iloc[j] and train['epoch'].iloc[i] <= gt['end_epoch'].iloc[j]:
        #         # generate the mask alpha
        #         mask = np.zeros(len(train.columns))
        #         mask


    

    


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

    # # print(train_df.values.shape)
    # # print(test_df.values.shape)


    # train_df.to_csv('./train.csv')
    # test_df.to_csv('./test.csv')

    # f = open('./list.txt', 'w')
    # for col in train.columns:
    #     f.write(col+'\n')
    # f.close()
    
# if __name__ == "__main__":
#     main()
