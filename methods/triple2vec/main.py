import pandas as pd
import os
import json
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
from config import DATA_DIR
# Prefer GPU 0 by default if available; can be overridden by env before launch
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
from embedding.triple2vec import triple2vec
from recommendation.recommender import triple2vecRec

def load_default_parameters(dataset_name):
    """
    Load default hyperparameters for the given dataset from best_params_config.json.

    """

    config_file = 'best_params_config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config[dataset_name]["best_parameters"]

def load_data(dataset_name, fold_id=0):
    history_data_path = f'../../datasets/{dataset_name}/history.json'
    with open(history_data_path, 'r') as f:
        history_data = json.load(f)
    records = []
    for user, baskets in history_data.items():
        for basket_idx, basket in enumerate(baskets[1:-1]):
            records.append({
                "UID": int(user),
                "PID": basket,
                "TS": basket_idx
            })
    df = pd.DataFrame(records)
    df['flag'] = 'train'
    future_data_path = f'../../datasets/{dataset_name}/future.json'
    with open(future_data_path, 'r') as f:
        future_data = json.load(f)
    records = []
    for user, baskets in future_data.items():
        for basket_idx, basket in enumerate(baskets[1:-1]):
            records.append({
                "UID": int(user),
                "PID": basket,
                "TS": basket_idx
            })

    keyset_path = f'../../datasets/{dataset_name}/keyset_{fold_id}.json'
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)
    train_users = [int(user) for user in keyset['train']]
    val_users = [int(user) for user in keyset['val']]
    test_users = [int(user) for user in keyset['test']]

    df_test = pd.DataFrame(records)
    # filter val users
    df_test.loc[df_test['UID'].isin(val_users), 'flag'] = 'validation'
    # filter test users
    df_test.loc[df_test['UID'].isin(test_users), 'flag'] = 'test'
    df_test = df_test.loc[df_test['flag'].isin(['validation', 'test'])]
    df = pd.concat([df, df_test])
    return df, keyset['item_num'], len(history_data)


def run_embedding(DATA_NAME, dim, lr, batch_size, n_neg, fold_id=0,
                  optimizer='momentum', max_epoch=500, n_sample=5000000,
                  n_sample_per_epoch=None, patience=5):

    myTrans, n_item, n_user = load_data(DATA_NAME, fold_id)
    dataTrain = myTrans[['UID', 'PID']].loc[myTrans['flag'] == 'train'].values
    
    embeddingDict = None
    myModel = triple2vec(DATA_NAME=DATA_NAME, 
                         HIDDEN_DIM=dim, LEARNING_RATE=lr, BATCH_SIZE=batch_size, 
                         N_NEG=n_neg, MAX_EPOCH=max_epoch, N_SAMPLE_PER_EPOCH=n_sample_per_epoch, PATIENCE=patience)
    # set fold for saving paths
    myModel.set_fold_id(fold_id)
    myModel.assign(dataTrain, n_user, n_item, N_SAMPLE=n_sample, dump=False)
    #myModel.assign_from_file(n_user, n_item)
    myModel.train(opt=optimizer)
    embeddingDict = myModel.extract_emebdding(True)
    return embeddingDict


def run_recommendation(data_name, dim=None, lr=None, batch_size=None, n_neg=None, l0=None, fold_id=0, top_k=10, ensemble=False):

    myTrans, n_item, n_user = load_data(data_name, fold_id)
    dataTrain = myTrans[['UID', 'PID']].loc[myTrans['flag'] == 'train'].values
    dataValidation = myTrans[['UID', 'PID']].loc[myTrans['flag'] == 'validation'].values
    dataTest = myTrans[['UID', 'PID']].loc[myTrans['flag'] == 'test'].values
    params = [data_name, 'triple2vec', dim, lr, batch_size, n_neg, fold_id]
    model_name = "_".join([str(p) for p in params])
    myRec = triple2vecRec(data_name, model_name, l0=l0, ensemble=ensemble, fold_id=fold_id)
    myRec.assign_data(dataTrain, dataValidation, dataTest, n_user, n_item)
    myRec.assign_embeddings()
    myRec.export_predictions_frequency_style(data_name, fold_id, top_k=top_k)
    resVali, resTest = myRec.evaluate(dump=False)
    

def main(argv):
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

    dataset_name = argv[1]
    default_params = load_default_parameters(dataset_name)  
    fold_id = int(argv[2]) if len(argv) > 2 else 0
    mode = argv[3] if len(argv) > 3 else "train"
    dim = int(argv[4]) if len(argv) > 4 else default_params.get('dim', 128)
    lr = float(argv[5]) if len(argv) > 5 else default_params.get('lr', 0.001)
    batch_size = int(argv[6]) if len(argv) > 6 else default_params.get('batch_size', 4096)
    n_neg = int(argv[7]) if len(argv) > 7 else default_params.get('n_neg', 25)
    optimizer = argv[8] if len(argv) > 8 else default_params.get('optimizer', 'adam')
    max_epoch = int(argv[9]) if len(argv) > 9 else default_params.get('max_epoch', 60)
    n_sample = int(argv[10]) if len(argv) > 10 else default_params.get('n_sample', 6000000)
    n_sample_per_epoch = int(argv[11]) if len(argv) > 11 else default_params.get('n_sample_per_epoch', None)
    patience = int(argv[12]) if len(argv) > 12 else default_params.get('patience', 5)
    l0 = float(argv[14]) if len(argv) > 14 else default_params.get('l0', -1)
    ensemble = (str(argv[13]).lower() in ('1', 'true', 'yes', 'y')) if len(argv) > 13 else default_params.get('ensemble', False)

    topk = 10

    # Print parameter information
    print(f"Dataset: {dataset_name}")
    print(f"Keyset Index: {fold_id}")
    print("Hyperparameters:")
    print(f"  dim: {dim}")
    print(f"  lr: {lr}")
    print(f"  batch_size: {batch_size}")
    print(f"  n_neg: {n_neg}")
    print(f"  optimizer: {optimizer}")
    print(f"  max_epoch: {max_epoch}")
    print(f"  n_sample: {n_sample}")
    print(f"  n_sample_per_epoch: {n_sample_per_epoch}")
    print(f"  patience: {patience}")
    print(f"  l0: {l0}")
    print(f"  ensemble: {ensemble}")

    if mode == 'train':
        run_embedding(
            dataset_name, dim, lr, batch_size, n_neg, fold_id,
            optimizer=optimizer, max_epoch=max_epoch, n_sample=n_sample,
            n_sample_per_epoch=n_sample_per_epoch, patience=patience
        )

    if l0 < 0:
        l0 = None
    run_recommendation(dataset_name, dim, lr, batch_size, n_neg, l0, fold_id, top_k=topk, ensemble=ensemble)

if __name__ == '__main__':
    main(sys.argv)