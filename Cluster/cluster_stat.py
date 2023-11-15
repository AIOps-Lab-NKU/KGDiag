import os
import numpy as np
import pandas as pd
import yaml
import sys

# cluster results process: 
# every case in GAIA lasts for 10 mins and every minute has a result.
# Merge every 10-minute results into 1 result based on labels and results.
# Rules:
# 0 is normal; 1-4 is failure types
# When label is 1-4, the result after merging is determined by the type (1-4) that occurs most frequently in 10 results. 
# If multiple types occur the same number of times, the type closest to the middle of the 10 mins is result.
# when label is 0, number of 0 in 10 results should exceed a specific threshold to consider merged result as normal(0). 
# Otherwise, it should be considered as a failure. The way to determine failure type can be seen as above.
# Attention:
# normal periods in GAIA are not multiples of 10. An extra 0 should be added.

def find_max(values,results):
    # ignore zero
    max_value = max(values[1:])
    if max_value == 0:
        print("Embedding is invalid. Please run main.py again.")
        sys.exit()
    max_index = []
    for i in range(1,len(values)):
        if values[i] == max_value:
            max_index.append(i)
    if len(max_index)==1:
        return max_index[0]
    # Priority is given to the type closest to the middle of the failure period.
    else:
        idx_min = []
        for m in max_index:
            idx_list = [i for i,x in enumerate(results) if x == m]
            min_idx_list = [abs(i-len(results)//2) for i in idx_list]
            idx_min.append(min(min_idx_list))
        
        return max_index[idx_min.index(min(idx_min))]


def cluster_stat(output_data_path,stat_output_path):
    normal_threshold = 6
    result = pd.read_csv(output_data_path,sep=',')
    
    # valid:
    if result['result'].tolist().count(0)==len(result['result'].tolist()):
        print("All results are ZERO! Embedding is invalid. Please run main.py again.")
        sys.exit()

    # add one row of zero:
    df = pd.DataFrame(np.insert(result.values, 2, values=[0,0], axis=0))
    df.columns = result.columns

    group = df.groupby(df.index//10)
    label_n = []
    res_n = []
    for _, g in group:
        label = g['label'].tolist()[0]
        label_n.append(label)
        values = [0 for i in range(5)]
        for _,row in g.iterrows():
            values[row['result']]+=1
        if label==0:
            if values[0]>=normal_threshold:
                res_n.append(0)
            else:
                res_n.append(find_max(values,g['result'].tolist()))
        else:
            res_n.append(find_max(values,g['result'].tolist()))

    dff = pd.DataFrame()
    dff['label'] = label_n
    dff['result'] = res_n
    dff.to_csv(stat_output_path,sep=',',index=None)
        
    

