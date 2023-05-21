import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import torch
import torch.utils.data

class SWaTDataset(torch.utils.data.Dataset):
    def __init__(self, root=None):
        self.data = pd.read_csv('data/swat/swat_processed.csv')
        self.explanation = pd.read_csv('data/swat/swat_gt_exp.csv')
        

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

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))


    d_data = d_data.transpose()

    return d_data.tolist(), d_labels.tolist()

def process_gt():
    df = pd.read_csv('swat_gt.csv')
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
    return



def main():
    gt = process_gt()

    test = pd.read_csv('../../../data2/xjiae/hddds/swat/swat_test.csv', index_col=0)
    train = pd.read_csv('../../../data2/xjiae/hddds/swat/swat_train.csv', index_col=0)
    
   


    # test = test.iloc[:, 1:]
    # train = train.iloc[:, 1:]
    
    test['date'] = pd.to_datetime(test.index)
    test['epoch'] = test['date'].astype('int64')//1e9
     
    # breakpoint()

    
    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())
    train_labels = train.attack
    test_labels = test.attack
    train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])

    # print(len(test.columns),test.columns)
    # print(len(train.columns),train.columns)
    all_mask = []
    for i in range(len(train)):
        mask = np.zeros(train.shape[1])
        all_mask.append(mask)
        
    for i in range(len(test)):
        if test_labels[i] == 0:
            mask = np.zeros(test.shape[1]-2)
        else:
            attacked = look_up(gt, test, test.epoch.iloc[i])
            mask = np.zeros(test.shape[1]-2)

            mask[attacked] = 1
        all_mask.append(mask)

    # train = train.drop(columns = ['date', 'epoch'])
    test = test.drop(columns = ['date', 'epoch'])
    
    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    train['label'] = train_labels
    test['label'] = test_labels
    
    
    combined = pd.concat([train, test])
    combined.to_csv("swat_processed.csv")
    df = pd.DataFrame(all_mask)
    df.to_csv('swat_gt_exp.csv', index = False)
    
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
    
if __name__ == "__main__":
    main()