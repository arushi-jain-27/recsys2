import pandas as pd
import argparse
import json
import random
import os
import numpy as np
from datetime import datetime

def parse_date(date_str):
    """Parse date string to datetime object."""
    if isinstance(date_str, (int, float)):
        return date_str  # Already numeric (order number)
    
    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d")
    except:
        return datetime.strptime(str(date_str), "%m/%d/%Y")


def convert_dates_to_days_since_first(date_sequences):
    """Convert date sequences to days since user's first order.
    
    
    Args:
        date_sequences: List of date lists (one per order/basket)
        
    Returns:
        List of numeric lists (days since first order)
    """
    # Parse all dates
    parsed_sequences = []
    all_dates = []
    
    for date_list in date_sequences:
        parsed_list = []
        for date in date_list:
            parsed = parse_date(date)
            parsed_list.append(parsed)
            if parsed is not None and not isinstance(parsed, (int, float)):
                all_dates.append(parsed)
        parsed_sequences.append(parsed_list)
    
    # If we have datetime objects, convert to days since min_date
    if all_dates and isinstance(all_dates[0], datetime):
        min_date = min(all_dates)
        result_sequences = []
        for parsed_list in parsed_sequences:
            result_list = []
            for parsed in parsed_list:
                if isinstance(parsed, datetime):
                    days = (parsed - min_date).days
                    result_list.append(days)
                elif parsed is not None:
                    result_list.append(parsed)
                else:
                    result_list.append(0)
            result_sequences.append(result_list)
        return result_sequences
    else:
        # Already numeric or failed to parse - return as is
        return [[d if d is not None else 0 for d in seq] for seq in parsed_sequences]


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
    user_prices = {}
    user_dates = {}
    
    for user_id, user_data in df.groupby('user_id'):
        remapped_user_id = user_id_mapping[user_id]
        user_orders[remapped_user_id] = []
        user_prices[remapped_user_id] = []
        user_dates[remapped_user_id] = []
        # For each user, get their orders sorted by order_id
        for order_id, order_data in user_data.groupby('order_id'):
            # Remap product IDs to consecutive integers
            products_in_order = [product_id_mapping[pid] for pid in order_data['product_id'].tolist()]
            user_orders[remapped_user_id].append(products_in_order)
            # Store prices for each product in the order
            if 'total_price' in df.columns:
                # Replace NaN values with 0.0 (or use fillna)
                prices_in_order = order_data['total_price'].fillna(0.0).tolist()
                user_prices[remapped_user_id].append(prices_in_order)
            else:
                # If no price column, use 1.0 as default
                prices_in_order = [1.0] * len(products_in_order)
                user_prices[remapped_user_id].append(prices_in_order)
            # Store dates for each product in the order
            if 'date' in df.columns:
                dates_in_order = order_data['date'].tolist()
                user_dates[remapped_user_id].append(dates_in_order)
            else:
                # If no date column, use order sequence numbers (0, 1, 2, ...)
                dates_in_order = [len(user_orders[remapped_user_id]) - 1] * len(products_in_order)
                user_dates[remapped_user_id].append(dates_in_order)
    
    history_data = {}
    future_data = {}
    prices_data = {}
    dates_data = {}
    
    for user_id, order_sequences in user_orders.items():
        
        if len(order_sequences) < 2:
            # Skip users with less than 2 orders or more than 100 orders
            continue
            
        # History: last 100 orders except the last one, with [-1] markers
        # Take only the last 100 orders (or all if less than 100)
        # recent_orders = order_sequences[-100:] if len(order_sequences) > 100 else order_sequences
        history_sequence = [[-1]] + order_sequences[:-1] + [[-1]]
        history_data[str(user_id)] = history_sequence
        
        # Future: only the last order, with [-1] markers
        future_sequence = [[-1]] + [order_sequences[-1]] + [[-1]]
        future_data[str(user_id)] = future_sequence
        
        # Prices: corresponding prices for history orders (exclude last order), with [-1] markers
        price_sequences = user_prices[user_id]
        prices_sequence = [[-1]] + price_sequences[:-1] + [[-1]]
        prices_data[str(user_id)] = prices_sequence
        
        # Dates: corresponding dates for history orders (exclude last order), with [-1] markers
        date_sequences = user_dates[user_id]
        date_sequences_history = date_sequences[:-1]  # Exclude last order for history
        
        # Convert to days since first order
        converted_dates = convert_dates_to_days_since_first(date_sequences_history)
        
        dates_sequence = [[-1]] + converted_dates + [[-1]]
        dates_data[str(user_id)] = dates_sequence
    
    # Save history data
 
    with open(f'{dataset_name}/history.json', 'w') as f:
        json.dump(history_data, f)
    
    # Save future data
    with open(f'{dataset_name}/future.json', 'w') as f:
        json.dump(future_data, f)
    
    # Save prices data
    with open(f'{dataset_name}/prices.json', 'w') as f:
        json.dump(prices_data, f)
    
    # Save dates data
    with open(f'{dataset_name}/dates.json', 'w') as f:
        json.dump(dates_data, f)
    
    print(f"Created {dataset_name}/history.json with {len(history_data)} users")
    print(f"Created {dataset_name}/future.json with {len(future_data)} users")
    print(f"Created {dataset_name}/prices.json with {len(prices_data)} users")
    print(f"Created {dataset_name}/dates.json with {len(dates_data)} users")
    
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