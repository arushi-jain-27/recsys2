import pandas as pd
import argparse
import json
import random


def preprocess(dataset_name):

    df = pd.read_csv(f"{dataset_name}/full.csv")
    
    # Create mappings for user and product IDs to consecutive integers
    unique_users = df['user_id'].unique()
    unique_products = df['product_id'].unique()
    
    # Create user ID mapping: original_id -> new_id (0 to num_users-1)
    user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    # Create product ID mapping: original_id -> new_id (0 to num_products-1)
    product_id_mapping = {product_id: idx for idx, product_id in enumerate(unique_products)}
    
    item_num = len(unique_products)
    user_num = len(unique_users)
    
    print(f"Total users: {user_num}")
    print(f"Total products: {item_num}")

    # Get all orders for each user with remapped IDs
    user_orders = {}
    
    for user_id, user_data in df.groupby('user_id'):
        remapped_user_id = user_id_mapping[user_id]
        user_orders[remapped_user_id] = []
        # For each user, get their orders sorted by order_id
        for order_id, order_data in user_data.groupby('order_id'):
            # Remap product IDs to consecutive integers
            products_in_order = [product_id_mapping[pid] for pid in order_data['product_id'].tolist()]
            user_orders[remapped_user_id].append(products_in_order)
    
    history_data = {}
    future_data = {}
    
    for user_id, order_sequences in user_orders.items():
        
        if len(order_sequences) < 2:
            # Skip users with less than 2 orders
            continue
            
        # History: all orders except the last one, with [-1] markers
        history_sequence = [[-1]] + order_sequences[:-1] + [[-1]]
        history_data[str(user_id)] = history_sequence
        
        # Future: only the last order, with [-1] markers
        future_sequence = [[-1]] + [order_sequences[-1]] + [[-1]]
        future_data[str(user_id)] = future_sequence
    
    with open(f'{dataset_name}/history.json', 'w') as f:
        json.dump(history_data, f)
    
    with open(f'{dataset_name}/future.json', 'w') as f:
        json.dump(future_data, f)
    
    print(f"Created {dataset_name}/history.json with {len(history_data)} users")
    print(f"Created {dataset_name}/future.json with {len(future_data)} users")
    
    # Create keyset splits for train/validation/test
    for i in range(3):
        create_keyset_splits(dataset_name, history_data, future_data, item_num, i)

def create_keyset_splits(dataset_name, history_data, future_data, item_num, fold_id):
    """
    Create train/validation/test splits of users and save as keyset file.
    Based on the logic from keyset_fold.py
    """
    # Get all unique users (those who appear in both history and future)
    users = list(history_data.keys())
    user_num = len(users)
    
    # Shuffle users randomly with reproducible seed
    random.seed(fold_id)
    random.shuffle(users)
    
    # Calculate split sizes (same ratios as keyset_fold.py)
    # 80% for training, 20% for testing
    # Within training: 90% train, 10% validation
    train_user = users[:int(user_num * 4/5 * 0.9)]
    val_user = users[int(user_num * 4/5 * 0.9):int(user_num * 4/5)]
    test_user = users[int(user_num * 4/5):]
    
    
    # Create keyset dictionary
    keyset_dict = dict()
    keyset_dict['item_num'] = item_num
    keyset_dict['train'] = train_user
    keyset_dict['val'] = val_user
    keyset_dict['test'] = test_user
    
    print(f"Keyset {fold_id} splits: {len(train_user)} train, {len(val_user)} val, {len(test_user)} test users")
    print(f"Total items: {item_num}")
    

    
    # Save keyset file (default fold_id = 0)
    keyset_file = f'{dataset_name}/keyset_{fold_id}.json'
    with open(keyset_file, 'w') as f:
        json.dump(keyset_dict, f)
    
    print(f"Created {keyset_file}")

def main(dataset_name):
    print(f"Preprocessing dataset: {dataset_name}")
    
    preprocess(dataset_name)
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess recommendation datasets')
    parser.add_argument('dataset_name', help='Name of the dataset to preprocess (e.g., dunnhumby, instacart)')
    
    args = parser.parse_args()
    main(args.dataset_name)
