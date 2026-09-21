from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "dunnhumby",
    "instacart",
    "retailrocket",
    "sam",
    "taobao",
    "valuedshopper",
    "tmall",
    "tafeng",
]


def jaccard(a, b):
    return len(a & b) / len(a | b)


def compute_features(baskets):
    baskets = [set(basket) for basket in baskets]

    n_baskets = len(baskets)
    basket_sizes = [len(b) for b in baskets]

    all_items = [item for basket in baskets for item in basket]
    counts = Counter(all_items)

    n_purchases = len(all_items)
    n_unique = len(counts)

    avg_basket_size = np.mean(basket_sizes)
    repeat_rate = 1 - n_unique / n_purchases

    if n_baskets >= 2:
        adjacent_basket_jaccard = np.mean([
            jaccard(baskets[t], baskets[t + 1])
            for t in range(n_baskets - 1)
        ])
    else:
        adjacent_basket_jaccard = np.nan

    if n_baskets >= 2:
        previous_items = set().union(*baskets[:-1])
        last_basket = baskets[-1]
        recent_basket_novelty = (
            len(last_basket - previous_items) / len(last_basket)
        )
    else:
        recent_basket_novelty = np.nan

    return {
        "n_baskets": n_baskets,
        "avg_basket_size": avg_basket_size,
        "repeat_rate": repeat_rate,
        "adjacent_basket_jaccard": adjacent_basket_jaccard,
        "recent_basket_novelty": recent_basket_novelty,
    }


rows = []

for dataset in DATASETS:
    history_path = ROOT / "datasets" / dataset / "history.json"

    with open(history_path) as f:
        history = json.load(f)

    for user, stored_baskets in history.items():

        baskets = stored_baskets[1:-1]

        features = compute_features(baskets)

        rows.append({
            "dataset": dataset,
            "user": user,
            **features,
        })


features = pd.DataFrame(rows)

output_path = ROOT / "analysis" / "data" / "user_history_features.csv"
features.to_csv(output_path, index=False)


print("\nFeature table shape:")
print(features.shape)

print("\nUsers by dataset:")
print(features.groupby("dataset").size())

print("\nMissing values:")
print(features.isna().sum())

print("\nDataset-level feature means:")
print(
    features.groupby("dataset")
    .mean(numeric_only=True)
    .round(3)
)

print(f"\nSaved to {output_path}")
