import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.utils.data

class WADIDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, all = False, train = False, raw = False):
        if all:
            train_data = pd.read_csv('data/wadi/train_processed.csv')
            test_data = pd.read_csv('data/wadi/test_processed.csv')
            self.data = pd.concat([train_data, test_data])
            train_explanation = pd.DataFrame(np.zeros((train_data.shape[0], train_data.shape[1]-3)))
            test_explanation = pd.read_csv('data/wadi/test_gt_exp.csv')
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        else:
            if train:
                self.data = pd.read_csv('data/wadi/train_processed.csv')
                self.explanation = pd.DataFrame(np.zeros((self.data.shape[0], self.data.shape[1]-3)))
            else:
                self.data = pd.read_csv('data/wadi/test_processed.csv')
                self.explanation = pd.read_csv('data/wadi/test_gt_exp.csv')
        if raw:
            self.data = pd.read_csv('data/wadi/raw.csv')   
            test_explanation = pd.read_csv('data/wadi/test_gt_exp.csv')
            train_explanation = pd.DataFrame(np.zeros((self.data.shape[0]-len(test_explanation), test_explanation.shape[1])))
            self.explanation = pd.DataFrame(np.vstack([train_explanation.values, test_explanation.values]), columns=test_explanation.columns)
        self.timestamp = self.data['epoch']
        self.label = self.data['label']
     
        

    def __getitem__(self, index):
        return self.data.iloc[index, 1:-2].values, self.data.iloc[index, -2], self.explanation.iloc[index, :].values
            
       

    def __len__(self):
        return self.data.shape[0]
    
    
class WADISlidingDataset(torch.utils.data.Dataset):
    def __init__(self, window_size, stride=1, train = True):
        if train:
            df =  pd.read_csv('data/wadi/train_processed.csv', index_col=0)
            self.explanation = pd.DataFrame(np.zeros((df.shape[0], df.shape[1]-2)))
        else:
            df = pd.read_csv('data/wadi/test_processed.csv', index_col=0)
            self.explanation = pd.read_csv('data/wadi/test_gt_exp.csv')
            attacks = df['label'].values
        
      
        self.ts = df['epoch'].values
        self.window_size = window_size
        df_tag = df.drop(columns=['epoch', 'label'])
        self.tag_values = np.array(df_tag, dtype=np.float32)
        self.valid_idxs = []
        for L in range(len(self.ts) - self.window_size + 1):
            R = L + self.window_size - 1
            if (self.ts[R]-self.ts[L]) == self.window_size - 1:
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")
        self.labels = df['label']
        self.num_features = df.shape[1] - 2
    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + self.window_size - 1
        WINDOW_GIVEN = self.window_size - 1
        item = {}
        item['y'] = 0
        idx = last
        if 1 in self.labels[i : i + WINDOW_GIVEN].values:
            item['y'] = 1
            idx = np.where(self.labels[i : i + WINDOW_GIVEN] == 1)[0][0] + last
        item["ts"] = self.ts[i + self.window_size - 1]
        item["x"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
        item["xl"] = torch.from_numpy(self.tag_values[last])
        # item['exp'] = torch.from_numpy(self.explanation.iloc[idx].values)
        item['exp'] = torch.from_numpy(self.explanation.iloc[i : i + WINDOW_GIVEN].values)
        return item
    
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
    df = pd.read_csv('../data/wadi/gt.csv')
    # f = lambda x: len(x.iloc[:,0].split(" "))-1
    # df['date'] = df.iloc[:,0].str.split(" ").str[0]
    # df['end_date'] = df.date.astype(str)+' '+ df.iloc[:,1]
    df['end_epoch'] = pd.to_datetime(df['end']).astype('int64')//1e9
    df['start_epoch'] = pd.to_datetime(df['start']).astype('int64')//1e9    
    return df

def look_up(gt, data, time):
    
    for j in range(len(gt)):
        if gt.start_epoch.iloc[j] <= time and gt.end_epoch.iloc[j] >= time:
            attacked = gt['attacked'].iloc[j]
          
            if ',' in attacked:
                attacked = attacked.split(", ")
                
            else:
                attacked = [attacked]
            mask = np.zeros(data.shape[1]-1)
            # breakpoint()
            # attacked_index = np.where(data.columns.isin(attacked))[0]
            sensors = data.columns
            for i in range(len(sensors)):
                for j in range(len(attacked)):
                    if attacked[j] in sensors[i]:
                        mask[i] = 1
            # if mask.sum()  > 1:             
            #     breakpoint()
       
            return 1, mask
    return 0, np.zeros(data.shape[1]-1)    
    
def preprocess(df):
    # Convert "Date" column to datetime object
    df["Date"] = pd.to_datetime(df["Date"])

    # Extract the time part from "Time" column
    df["Time"] = pd.to_datetime(df["Time"]).dt.time
    # Combine "Date" and "Time" columns
    combined_datetime = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))

    
    df = df.drop(columns = ['Date', 'Time'])
    cols = [x[46:] for x in df.columns] # remove column name prefixes
    df.columns = cols
    df = df.fillna(df.mean())
    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip())
    # Convert combined datetime column to epoch time
    df["epoch"] = combined_datetime.astype('int64')//1e9
    return df
def normalize(df, min, max):
        ndf = df.copy()
        for c in df.columns:
            if min[c] == max[c]:
                ndf[c] = df[c] - min[c]
            else:
                ndf[c] = (df[c] - min[c]) / (max[c] - min[c])
        return ndf
def main():
    gt = process_gt()
    # infile = open('../../../data2/xjiae/hddds/wadi/WADI_14days.csv', 'r')
    # for i in range(5):
    #     firstLine = infile.readline()
    #     print(firstLine)
    #     breakpoint()

    train = pd.read_csv('../../../../data2/xjiae/hddds/wadi/WADI_14days.csv', index_col=0, skiprows=4)

    
    test = pd.read_csv('../../../../data2/xjiae/hddds/wadi/WADI_attackdata.csv', index_col=0)
    
    
    train = preprocess(train)

    test = preprocess(test)
    
    all_mask = []
    # for i in tqdm(range(len(train))):
    #     mask = np.zeros(train.shape[1]-1)
    #     all_mask.append(mask)
    train_label = np.zeros(train.shape[0])

    test_label = []
    
    for i in tqdm(range(len(test))):
        label, mask = look_up(gt, test, test.epoch.iloc[i])
        test_label.append(label)
        all_mask.append(mask)
    
    
    train_df = train.fillna(train.mean())
    test_df = test.fillna(test.mean())
    
    test_df['label'] = test_label
    train_df['label'] = train_label
    
    # combined = pd.concat([train_df, test_df])
    # combined.to_csv("../data/wadi/raw.csv")

    
    # test = test.drop(columns = ['epoch'])
    # train = train.drop(columns = ['epoch'])
    
    # train['label'] =  train_label
    # test['label'] = test_label
    
    # combined = pd.concat([train, test])
    # combined.to_csv("data/wadi/processed.csv")
    
    df = pd.DataFrame(all_mask)
    df.to_csv('../data/wadi/test_gt_exp.csv', index = False)
    # breakpoint()
    

    train_df = train_df.drop(columns=['label', 'epoch'])
    test_df = test_df.drop(columns=['label', 'epoch'])
    
    min = train_df.min()
    max = train_df.max()  
    
    
    train_df = normalize(train_df, min, max).ewm(alpha=0.9).mean()
    test_df = normalize(test_df, min, max).ewm(alpha=0.9).mean()
    
    
    train_df['label'] = train_label
    train_df['epoch'] = train['epoch']
    
    
    test_df['label'] = test_label
    test_df['epoch'] = test['epoch']
        
    train_df.to_csv("../data/wadi/train_processed.csv")
    test_df.to_csv("../data/wadi/test_processed.csv")
    
    
    breakpoint()
        
    
    
        

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