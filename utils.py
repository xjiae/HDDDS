from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
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

def threshold_array(arr, threshold):
    # Create a copy of the input array
    transformed_arr = np.copy(arr)
    
    # Set values larger than the threshold to 1, and all others to 0
    transformed_arr[transformed_arr > threshold] = 1
    transformed_arr[transformed_arr <= threshold] = 0
    
    return transformed_arr
def summary(y_true, y_pred, score = True):
  
    if score:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        recall = recall + 1e-10
        f1_scores = 2*recall*precision/(recall+precision)
        threshold = thresholds[np.argmax(f1_scores)]
        
        
        y_pred = threshold_array(y_pred, threshold)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print(f"& {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\")
    return acc, f1, fpr, fnr