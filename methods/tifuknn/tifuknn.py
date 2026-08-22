from __future__ import unicode_literals, print_function, division
import numpy as np
import sys
import math
import csv

import os
import json

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

topk = 10

def group_history_list(his_list, num_groups):
    """
    Group user's basket history into time periods and average each group.
    
    Args:
        his_list: List of historical baskets (vectors)
        num_groups: Number of time groups to create
        
    Returns:
        grouped_vec_list: List of averaged vectors for each time group
        real_num_groups: Actual number of groups created
    """
    if not his_list:
        return [], 0
    
    # If fewer baskets than groups, return each basket as its own group
    if len(his_list) <= num_groups:
        return his_list, len(his_list)
    
    # Calculate distribution (matching original algorithm exactly)
    base_baskets_per_group = len(his_list) // num_groups
    extra_baskets = len(his_list) % num_groups
    
    grouped_vec_list = []
    basket_idx = 0
    
    # First groups get base size
    groups_with_base_size = num_groups - extra_baskets
    for group_idx in range(groups_with_base_size):
        group_sum = np.zeros(len(his_list[0]))
        for _ in range(base_baskets_per_group):
            group_sum += his_list[basket_idx]
            basket_idx += 1
        grouped_vec_list.append(group_sum / base_baskets_per_group)
    
    # Last groups get base size + 1 (the extra baskets)
    baskets_per_extra_group = base_baskets_per_group + 1
    for group_idx in range(groups_with_base_size, num_groups):
        group_sum = np.zeros(len(his_list[0]))
        for _ in range(baskets_per_extra_group):
            group_sum += his_list[basket_idx]
            basket_idx += 1
        grouped_vec_list.append(group_sum / baskets_per_extra_group)
    
    return grouped_vec_list, num_groups

def temporal_decay_sum_history(data_set, key_set, output_size, num_groups, within_decay_rate, group_decay_rate):
    sum_history = {}
    for key in key_set:
        vec_list = data_set[key] # basket list
        num_vec = len(vec_list) - 2
        his_list = []
        for idx in range(1,num_vec+1):
            his_vec = np.zeros(output_size)
            decayed_val = np.power(within_decay_rate, num_vec-idx)
            for ele in vec_list[idx]:
                his_vec[ele] = decayed_val
            his_list.append(his_vec)

        grouped_list, real_num_groups = group_history_list(his_list, num_groups)
        his_vec = np.zeros(output_size)
        for idx in range(real_num_groups):
            decayed_val = np.power(group_decay_rate, real_num_groups - 1 - idx)
            his_vec += grouped_list[idx]*decayed_val
        sum_history[key] = his_vec/real_num_groups
    return sum_history

def KNN(query_set, target_set, k):
    history_mat = []
    for key in target_set.keys():
        history_mat.append(target_set[key])
    test_mat = []
    for key in query_set.keys():
        test_mat.append(query_set[key])
    # print('Finding k nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(history_mat)
    distances, indices = nbrs.kneighbors(test_mat)
    # print('Finish KNN search.' )
    return indices, distances

def vec2label_list(pred_vec):
    label_list = pred_vec.argsort()[::-1][:topk].tolist()
    return label_list

def merge_history(sum_history_test, test_key_set, training_sum_history_test, training_key_set, index, alpha):
    merged_history = {}
    for test_key_id in range(len(test_key_set)):
        test_key = test_key_set[test_key_id]
        test_history = sum_history_test[test_key]
        sum_training_history = np.zeros(len(test_history))
        for indecis in index[test_key_id]:
            training_key = training_key_set[indecis]
            sum_training_history += training_sum_history_test[training_key]

        sum_training_history = sum_training_history/len(index[test_key_id])

        merge = test_history*alpha + sum_training_history*(1-alpha)
        merge = vec2label_list(merge) #transfer to label list
        merged_history[test_key] = merge

    return merged_history

def evaluate(data_history, training_key_set, test_key_set, input_size, num_groups,
             within_decay_rate, group_decay_rate, num_nearest_neighbors, alpha):

    temporal_decay_sum_history_training = temporal_decay_sum_history(data_history,
                                                                     training_key_set, input_size,
                                                                     num_groups, within_decay_rate,
                                                                     group_decay_rate) #don't need to change

    temporal_decay_sum_history_test = temporal_decay_sum_history(data_history,
                                                                 test_key_set, input_size,
                                                                 num_groups, within_decay_rate,
                                                                 group_decay_rate)

    neighbour_index, distance = KNN(temporal_decay_sum_history_test, temporal_decay_sum_history_training,
                          num_nearest_neighbors)

    sum_history = merge_history(temporal_decay_sum_history_test, test_key_set, temporal_decay_sum_history_training,
                                training_key_set, neighbour_index, alpha)

    return sum_history

def load_default_parameters(dataset_name):
    """
    Load default hyperparameters from best_params_config.json for the given dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        dict: Dictionary containing the hyperparameters
    """
    config_file = "best_params_config.json"
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config[dataset_name]['best_parameters']  
     
def main(argv):
    # param setting
    dataset_name = argv[1]
    
    # Load default parameters from config file
    default_params = load_default_parameters(dataset_name)
    
    # Parse command line arguments (optional overrides)
    # Format: python tifuknn.py dataset_name [keyset_index] [num_nearest_neighbors] [within_decay_rate] [group_decay_rate] [alpha] [num_groups]
    keyset_index = int(argv[2]) if len(argv) > 2 else 0
    num_nearest_neighbors = int(argv[3]) if len(argv) > 3 else default_params["num_nearest_neighbors"]
    within_decay_rate = float(argv[4]) if len(argv) > 4 else default_params["within_decay_rate"]
    group_decay_rate = float(argv[5]) if len(argv) > 5 else default_params["group_decay_rate"]
    alpha = float(argv[6]) if len(argv) > 6 else default_params["alpha"]
    num_groups = int(argv[7]) if len(argv) > 7 else default_params["num_groups"]
    
    # Print parameter information
    print(f"Dataset: {dataset_name}")
    print(f"Keyset Index: {keyset_index}")
    print("Hyperparameters:")
    print(f"  num_nearest_neighbors: {num_nearest_neighbors}")
    print(f"  within_decay_rate: {within_decay_rate}")
    print(f"  group_decay_rate: {group_decay_rate}")
    print(f"  alpha: {alpha}")
    print(f"  num_groups: {num_groups}")
    
    history_file = f"../../datasets/{dataset_name}/history.json"
    keyset_file = f"../../datasets/{dataset_name}/keyset_{keyset_index}.json"


    with open(history_file, 'r') as f:
        data_history = json.load(f)
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    #vector size -> item num, get train meta test user list[]
    input_size = keyset['item_num']
    keyset_train = keyset['train']
    keyset_val = keyset['val']
    keyset_test = keyset['test']

    predicted_test = evaluate(data_history, keyset_train, keyset_test, input_size,
                                num_groups, within_decay_rate, group_decay_rate,
                                num_nearest_neighbors, alpha)

    predicted_val = evaluate(data_history, keyset_train, keyset_val, input_size,
                                num_groups, within_decay_rate, group_decay_rate,
                                num_nearest_neighbors, alpha)
    pred_dict = dict()
    pred_dict.update(predicted_val)
    pred_dict.update(predicted_test)

    pred_path = f"../../predictions/{dataset_name}/tifuknn/keyset{keyset_index}.json"
    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)


if __name__ == '__main__':
    main(sys.argv)
