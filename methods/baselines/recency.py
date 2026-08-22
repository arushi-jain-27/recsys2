import sys
import json
import os
from typing import Dict, List
from datetime import datetime
topk = 10
global_window = 50  # number of most recent baskets to consider per user
global_decay = 0.95  # geometric decay per basket distance from latest


def compute_global_recency_scores(data_history: Dict[str, List[List[int]]], window: int, decay: float) -> Dict[int, float]:
    """Compute a global recency score per item using the last `window` baskets of each user.

    More recent baskets contribute more via geometric decay: weight = decay**distance,
    where distance 0 is the latest basket.
    """
    scores: Dict[int, float] = {}
    for _, baskets in data_history.items():
        core_baskets = baskets[1:-1] if baskets else []
        if not core_baskets:
            continue
        # iterate last `window` baskets with distance from end
        end = len(core_baskets)
        start = max(0, end - window)
        for idx in range(start, end):
            distance = (end - 1) - idx  # 0 for latest
            weight = decay ** distance
            for item in core_baskets[idx]:
                if item == -1:
                    continue
                scores[item] = scores.get(item, 0.0) + weight
    return scores


def evaluate(data_history: Dict[str, List[List[int]]], key_set: List[str], top_k: int, global_scores: Dict[int, float]) -> Dict[str, List[int]]:
    predictions: Dict[str, List[int]] = {}
    for user_id in key_set:
        baskets = data_history.get(user_id, [])
        # Ignore padding [-1] at both ends
        core_baskets = baskets[1:-1] if baskets else []

        # Track most recent index each item appeared at (higher index = more recent)
        last_seen_index: Dict[int, int] = {}
        for idx, basket in enumerate(core_baskets):
            for item in basket:
                if item != -1:
                    last_seen_index[item] = idx

        # Sort by user recency (desc), then global recency score (desc), then item id (asc)
        ranked_items = sorted(
            last_seen_index.items(),
            key=lambda x: (-x[1], -global_scores.get(x[0], 0.0),  x[0])
        )
        predictions[user_id] = [item for item, _ in ranked_items[:top_k]]
    return predictions


def main(argv):
    dataset_name = argv[1]
    keyset_index = int(argv[2]) if len(argv) > 2 else 0

    history_file = f"../../datasets/{dataset_name}/history.json"
    keyset_file = f"../../datasets/{dataset_name}/keyset_{keyset_index}.json"

    with open(history_file, 'r') as f:
        data_history = json.load(f)
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    
    keyset_train = keyset['train']
    keyset_val = keyset['val']
    keyset_test = keyset['test']

    global_scores = compute_global_recency_scores(data_history, global_window, global_decay)

    predicted_test = evaluate(data_history, keyset_test, topk, global_scores)
    predicted_val = evaluate(data_history, keyset_val, topk, global_scores)

    pred_dict = dict()
    pred_dict.update(predicted_val)
    pred_dict.update(predicted_test)

    pred_path = f"../../predictions/{dataset_name}/recency/keyset{keyset_index}.json"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)

    print(f"{datetime.now()}: Recent Predictions saved to {pred_path}")



if __name__ == "__main__":
    main(sys.argv)


