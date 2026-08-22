import sys
import json
import os
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

topk = 10


def load_data(history_file: str, prices_file: str, dates_file: str):
    """Load history, prices, and dates data."""
    with open(history_file, "r") as f:
        history = json.load(f)
    with open(prices_file, "r") as f:
        prices = json.load(f)
    with open(dates_file, "r") as f:
        dates = json.load(f)
    return history, prices, dates


def compute_rfm_scores(user_id: str, baskets: List[List[int]], prices: List[List[float]], dates: List[List]) -> Dict[int, float]:
    """Compute RFM scores for items in a user's purchase history.
    
    Args:
        user_id: User identifier
        baskets: List of baskets (each basket is a list of item IDs)
        prices: List of price lists corresponding to each basket
        dates: List of date lists corresponding to each basket
        
    Returns:
        Dictionary mapping item_id to RFM score
    """
    # Ignore padding baskets [-1] at start and end
    core_baskets = baskets[1:-1] if len(baskets) > 2 else []
    core_prices = prices[1:-1] if len(prices) > 2 else []
    core_dates = dates[1:-1] if len(dates) > 2 else []
    
    if not core_baskets:
        return {}
    
    # Track metrics per item
    # Dates are already preprocessed as "days since user's first order"
    item_recency = {}  # Most recent date (days since first order) for each item
    item_frequency = defaultdict(int)  # Count of purchases
    item_revenue = defaultdict(float)  # Total price spent on item
    
    for basket_idx, (basket, price_list, date_list) in enumerate(zip(core_baskets, core_prices, core_dates)):
        for item, price, date in zip(basket, price_list, date_list):
            if item == -1:
                continue
            # Date is already numeric (days since first order, computed in preprocessing)
            # Keep the maximum (most recent) value for each item
            if item not in item_recency or date > item_recency[item]:
                item_recency[item] = date
            item_frequency[item] += 1
            item_revenue[item] += price
    
    if not item_recency:
        return {}
    
    # Normalize metrics within user
    def min_max_normalize(values_dict):
        """Min-max normalize a dictionary of values."""
        if not values_dict:
            return {}
        min_val = min(values_dict.values())
        max_val = max(values_dict.values())
        if min_val == max_val:
            return {k: 1.0 for k in values_dict.keys()}
        return {k: (v - min_val) / (max_val - min_val) for k, v in values_dict.items()}
    
    # Get total revenue per user for revenue share calculation
    total_revenue = sum(item_revenue.values())
    item_revenue_share = {item: revenue / total_revenue if total_revenue > 0 else 0.0 
                          for item, revenue in item_revenue.items()}
    
    # Normalize each metric
    normalized_recency = min_max_normalize(item_recency)
    normalized_frequency = min_max_normalize(dict(item_frequency))
    normalized_revenue = min_max_normalize(item_revenue_share)
    
    # Compute RFM score as average of normalized metrics
    rfm_scores = {}
    for item in item_recency.keys():
        rfm_scores[item] = (
            normalized_recency[item] + 
            normalized_frequency[item] + 
            normalized_revenue[item]
        ) / 3.0
    
    return rfm_scores


def compute_global_item_scores(history: Dict[str, List[List[int]]]) -> Dict[int, float]:
    """Compute global popularity scores for items across all users (for tie-breaking)."""
    global_item_freq = defaultdict(int)
    for user_id, baskets in history.items():
        core_baskets = baskets[1:-1] if len(baskets) > 2 else []
        for basket in core_baskets:
            for item in basket:
                if item != -1:
                    global_item_freq[item] += 1
    return dict(global_item_freq)


def evaluate(data_history: Dict[str, List[List[int]]], 
             data_prices: Dict[str, List[List[float]]],
             data_dates: Dict[str, List[List]],
             key_set: List[str], 
             top_k: int,
             global_scores: Dict[int, float]) -> Dict[str, List[int]]:
    """Generate top-k recommendations for users based on RFM scores."""
    predictions: Dict[str, List[int]] = {}
    
    for user_id in key_set:
        baskets = data_history.get(user_id, [])
        prices = data_prices.get(user_id, [])
        dates = data_dates.get(user_id, [])
        
        # Compute RFM scores for this user
        rfm_scores = compute_rfm_scores(user_id, baskets, prices, dates)
        
        if not rfm_scores:
            predictions[user_id] = []
            continue
        
        # Sort by RFM score (desc), then global popularity (desc), then item_id (asc)
        ranked_items = sorted(
            rfm_scores.items(),
            key=lambda x: (-x[1], -global_scores.get(x[0], 0), x[0])
        )
        
        predictions[user_id] = [item for item, _ in ranked_items[:top_k]]
    
    return predictions


def main(argv):
    dataset_name = argv[1]
    keyset_index = int(argv[2]) if len(argv) > 2 else 0

    history_file = f"../../datasets/{dataset_name}/history.json"
    prices_file = f"../../datasets/{dataset_name}/prices.json"
    dates_file = f"../../datasets/{dataset_name}/dates.json"
    keyset_file = f"../../datasets/{dataset_name}/keyset_{keyset_index}.json"

    # Load data
    data_history, data_prices, data_dates = load_data(history_file, prices_file, dates_file)
    
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    keyset_train = keyset['train']
    keyset_val = keyset['val']
    keyset_test = keyset['test']

    # Compute global item scores for tie-breaking
    global_scores = compute_global_item_scores(data_history)

    # Generate predictions
    predicted_test = evaluate(data_history, data_prices, data_dates, keyset_test, topk, global_scores)
    predicted_val = evaluate(data_history, data_prices, data_dates, keyset_val, topk, global_scores)
    
    # Combine predictions
    pred_dict = dict()
    pred_dict.update(predicted_val)
    pred_dict.update(predicted_test)

    # Save predictions
    pred_path = f"../../predictions/{dataset_name}/rfm/keyset{keyset_index}.json"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)
    
    print(f"{datetime.now()}: RFM Predictions saved to {pred_path}")


if __name__ == "__main__":
    main(sys.argv)

