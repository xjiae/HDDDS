from hddds import *
import numpy as np
from tqdm import tqdm

ann_cnt = 0
fea_cnt = 0
gt_ratio = []
ano_cnt = 0
fea_num = 0
total = 0
total_ann = 0
for cat in MVTEC_CATEGORIES:
    ds_name = f'mvtec_{cat}'
    dataset = get_dataset(ds_name)
    total += len(dataset)
    for i in tqdm(range(len(dataset))):

        x, y, w = dataset[i]
        fea_num = x.shape[1] *  x.shape[2]
        if y == 1:
            ano_cnt += 1
            fea_cnt += w.sum()
            gt_ratio.append(w.sum()/fea_num)
        
        
    
    
    
avg_fea = fea_cnt/ano_cnt
avg_gt_ratio = np.mean(gt_ratio)    


print(f"{fea_num} & {ano_cnt} & {total - ano_cnt} & {total} & {(ano_cnt/total)*100:.2f}\% \\\\")

print(f"{total * fea_num} & {avg_fea:.4f} & {avg_gt_ratio*100:.2f}\% \\\\")
breakpoint()
    