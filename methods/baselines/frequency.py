import sys
import json
from collections import Counter
from typing import Dict, List, Iterable
from datetime import datetime
import os
topk = 10

def load_history(history_file: str) -> Dict[str, List[List[int]]]:
    with open(history_file, "r") as f:
        return json.load(f)


def iter_user_items(baskets: List[List[int]]) -> Iterable[int]:
    # Ignore padding baskets equal to [-1]
    for basket in baskets:
        for item in basket:
            if item != -1:
                yield item


def compute_user_frequencies(history: Dict[str, List[List[int]]]) -> Dict[str, Counter]:
    user_to_counter: Dict[str, Counter] = {}
    for user_id, baskets in history.items():
        user_to_counter[user_id] = Counter(iter_user_items(baskets[1:-1]))
    return user_to_counter


def compute_global_frequencies(history: Dict[str, List[List[int]]]) -> Counter:
    global_counter: Counter = Counter()
    for _, baskets in history.items():
        global_counter.update(iter_user_items(baskets[1:-1]))
    return global_counter


def evaluate(data_history: Dict[str, List[List[int]]], key_set: List[str], top_k: int, global_counter: Counter) -> Dict[str, List[int]]:
    predictions: Dict[str, List[int]] = {}
    for user_id in key_set:
        baskets = data_history.get(user_id, [])
        counter = Counter(iter_user_items(baskets[1:-1])) if baskets else Counter()
        # Sort by user frequency desc, then global frequency desc, then item id asc
        ranked = sorted(
            counter.items(),
            key=lambda x: (-x[1], -global_counter.get(x[0], 0), x[0])
        )
        predictions[user_id] = [item for item, _ in ranked[:top_k]]
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

    global_counter = compute_global_frequencies(data_history)

    predicted_test = evaluate(data_history, keyset_test, topk, global_counter)
    predicted_val = evaluate(data_history, keyset_val, topk, global_counter)
    
    pred_dict = dict()
    pred_dict.update(predicted_val)
    pred_dict.update(predicted_test)

    pred_path = f"../../predictions/{dataset_name}/frequency/keyset{keyset_index}.json"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)
    print(f"{datetime.now()}: Popular Predictions saved to {pred_path}")


if __name__ == "__main__":
    main(sys.argv)


