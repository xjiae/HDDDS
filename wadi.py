import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.utils.data

class WADIDataset(torch.utils.data.Dataset):
    def __init__(self, root=None):
        self.data = pd.read_csv('data/wadi/processed.csv')
        self.explanation = pd.read_csv('data/wadi/gt_exp.csv')
        

    def __getitem__(self, index):
        return self.data.iloc[index, 1:-1].values, self.data.iloc[index, -1], self.explanation.iloc[index, :].values
       

    def __len__(self):
        return self.data.shape[0]

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
    df = pd.read_csv('data/wadi/gt.csv')
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

def main():
    gt = process_gt()
    # infile = open('../../../data2/xjiae/hddds/wadi/WADI_14days.csv', 'r')
    # for i in range(5):
    #     firstLine = infile.readline()
    #     print(firstLine)
    #     breakpoint()

    train = pd.read_csv('../../../data2/xjiae/hddds/wadi/WADI_14days.csv', index_col=0, skiprows=4)

    
    test = pd.read_csv('../../../data2/xjiae/hddds/wadi/WADI_attackdata.csv', index_col=0)
    
    
    train = preprocess(train)

    test = preprocess(test)
    
    all_mask = []
    for i in tqdm(range(len(train))):
        mask = np.zeros(train.shape[1]-1)
        all_mask.append(mask)
    train_label = np.zeros(train.shape[0])

    test_label = []
    
    for i in tqdm(range(len(test))):
        label, mask = look_up(gt, test, test.epoch.iloc[i])
        test_label.append(label)
        all_mask.append(mask)
    
    test = test.drop(columns = ['epoch'])
    train = train.drop(columns = ['epoch'])
    
    train['label'] =  train_label
    test['label'] = test_label
    
    combined = pd.concat([train, test])
    combined.to_csv("data/wadi/processed.csv")
    
    df = pd.DataFrame(all_mask)
    df.to_csv('data/wadi/gt_exp.csv', index = False)
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