import json
import numpy as np
from scipy import sparse
import os
import sys
import pandas as pd
from similarity import *


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


def data_parser(dataset_name):
    history_data_path = f'../../datasets/{dataset_name}/history.json'
    with open(history_data_path, 'r') as f:
        history_data = json.load(f)
    records = []
    baskets_count = 0
    for user, baskets in history_data.items():
        for basket_idx, basket in enumerate(baskets[1:-1]):
            for item in basket:
                records.append({
                    "UID": int(user),
                    "PID": item,
                    "BID": baskets_count,
                    "order": basket_idx
                })
            baskets_count += 1
    df = pd.DataFrame(records)
    return df


def build_user_item_matrix(df_train):
    """Build item-user sparse matrix from training interactions DataFrame."""
    if df_train.empty:
        return sparse.csc_matrix((0, 0))

    user_num = int(df_train.UID.max()) + 1
    item_num = int(df_train.PID.max()) + 1

    cf_user_list = df_train['UID'].to_numpy()
    cf_item_list = df_train['PID'].to_numpy()
    user_item_matrix = sparse.coo_matrix(
        (np.ones((len(cf_user_list),), dtype=int), (cf_user_list, cf_item_list)),
        shape=(user_num, item_num)
    )
    # Transpose to item-user as in original RACF flow
    return user_item_matrix.tocsc().T


def compute_rc_matrix(df_train, user_num, item_num, recency):
    """Compute recency-weighted user-item scores from last `recency` baskets."""
    rc_user_list, rc_item_list, rc_score_list = [], [], []
    for uid, df_user in df_train.groupby('UID'):
        orders = sorted(df_user['order'].unique().tolist())
        if recency > 0 and len(orders) > recency:
            recent_orders = orders[-recency:]
        else:
            recent_orders = orders
        if len(recent_orders) == 0:
            continue
        freq = {}
        for ord_idx in recent_orders:
            items_in_basket = df_user[df_user['order'] == ord_idx]['PID'].tolist()
            for item in items_in_basket:
                freq[item] = freq.get(item, 0) + 1
        for item_id, count in freq.items():
            rc_user_list.append(int(uid))
            rc_item_list.append(int(item_id))
            rc_score_list.append(count / float(len(recent_orders)))

    return sparse.coo_matrix((rc_score_list, (rc_user_list, rc_item_list)), shape=(user_num, item_num))


def compute_user_similarity(item_user_matrix, alpha, topK=100, shrink=1000):
    """Compute user-user similarity matrix using asymmetric cosine."""
    sim_cls = Compute_Similarity(item_user_matrix, shrink=shrink, asymmetric_alpha=alpha, similarity='asymmetric', topK=topK)
    usersim = sim_cls.compute_similarity().T
    usersim = usersim.tocsr()
    usersim.setdiag(1.0)
    return usersim


def topk_from_rec_matrix(rec_mat, user_idx, topk):
    """Extract top-k item indices for a given user from recommendation matrix."""
    row = rec_mat.getrow(user_idx).toarray().ravel() if hasattr(rec_mat, 'getrow') else np.asarray(rec_mat[user_idx]).ravel()
    return row.argsort()[::-1][:topk].tolist()


def main(argv):
    dataset_name = argv[1]
    default_params = load_default_parameters(dataset_name)

    keyset_index = int(argv[2]) if len(argv) > 2 else 0
    recency = int(argv[3]) if len(argv) > 3 else int(default_params['recency'])
    alpha = float(argv[4]) if len(argv) > 4 else float(default_params['asymmetry'])
    q = int(argv[5]) if len(argv) > 5 else int(default_params['locality'])

    # Print parameters (aligned style)
    print(f"Dataset: {dataset_name}")
    print(f"Keyset Index: {keyset_index}")
    print("Hyperparameters:")
    print(f"  recency: {recency}")
    print(f"  asymmetry: {alpha}")
    print(f"  locality: {q}")

    # Load keyset
    keyset_file = f"../../datasets/{dataset_name}/keyset_{keyset_index}.json"
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    # Prepare training dataframe (same parser structure as cfr.py)
    df_train = data_parser(dataset_name)

    # Show some dimensions
    print('=============')
    n_users = df_train.UID.nunique()
    n_items = df_train.PID.nunique()
    n_baskets = df_train.BID.nunique()
    print('n_users:', n_users)
    print('n_items:', n_items)
    print('n_baskets:', n_baskets)
    print('=============')

    # Matrix shapes require max index + 1 in case IDs are 0-based contiguous
    user_num = int(df_train.UID.max()) + 1 if not df_train.empty else 0
    item_num = int(df_train.PID.max()) + 1 if not df_train.empty else 0

    k = 10

 
    user_item_matrix = build_user_item_matrix(df_train)

    rc_matrix = compute_rc_matrix(df_train, user_num, item_num, recency)
    
    usersim = compute_user_similarity(user_item_matrix, alpha, topK=100, shrink=1000)

    # Recommendation scores: (usersim ^ q) * rc_matrix
    user_recommendations = usersim.power(q).dot(sparse.csr_matrix(rc_matrix))
    print("RACF")
    print("User's Recommendations matrix dim:", user_recommendations.shape)

    # Prepare prediction users (val + test)
    users_val = keyset.get('val', [])
    users_test = keyset.get('test', [])
    users_pred = list(users_val) + list(users_test)

    # Helper to extract top-k per user index from a (sparse) recommendation matrix
    def topk_from_rec_matrix(rec_mat, user_idx, topk):
        row = rec_mat.getrow(user_idx).toarray().ravel() if hasattr(rec_mat, 'getrow') else np.asarray(rec_mat[user_idx]).ravel()
        return row.argsort()[::-1][:topk].tolist()

    pred_racf = {}
    for uid in users_pred:
        pred_racf[uid] = topk_from_rec_matrix(user_recommendations, uid, k)

    # Write RACF predictions
    out_dir_racf = f"../../predictions/{dataset_name}/racf"
    os.makedirs(out_dir_racf, exist_ok=True)
    with open(f"{out_dir_racf}/keyset{keyset_index}.json", 'w') as f:
        json.dump(pred_racf, f)


if __name__ == '__main__':
    main(sys.argv)







