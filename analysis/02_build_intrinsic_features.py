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


def js_divergence(counter_a, counter_b):
    items = list(set(counter_a) | set(counter_b))

    p = np.array([counter_a[i] for i in items], dtype=float)
    q = np.array([counter_b[i] for i in items], dtype=float)

    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0]))
    kl_qm = np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0]))

    return 0.5 * (kl_pm + kl_qm)


def compute_features(baskets):
    # Basket semantics: presence/absence, not within-basket quantity
    baskets = [set(basket) for basket in baskets]

    n_baskets = len(baskets)
    basket_sizes = [len(b) for b in baskets]

    all_items = [item for basket in baskets for item in basket]
    counts = Counter(all_items)

    n_purchases = len(all_items)
    n_unique = len(counts)

    # 1. History depth
    avg_basket_size = np.mean(basket_sizes)

    # 2. Repetition
    repeat_rate = 1 - n_unique / n_purchases

    # 3. Preference diversity / concentration
    if n_unique == 1:
        item_entropy = 0.0
    else:
        probs = np.array(list(counts.values()), dtype=float) / n_purchases
        entropy = -np.sum(probs * np.log(probs))
        item_entropy = entropy / np.log(n_unique)

    # 4. One-off items
    singleton_item_share = (
        sum(count == 1 for count in counts.values()) / n_unique
    )

    # 5. Consecutive-basket persistence
    if n_baskets >= 2:
        adjacent_basket_jaccard = np.mean([
            jaccard(baskets[t], baskets[t + 1])
            for t in range(n_baskets - 1)
        ])
    else:
        adjacent_basket_jaccard = np.nan

    # 6. Recency vs frequency disagreement
    k = min(10, n_unique)

    frequent_items = {
        item
        for item, _ in sorted(
            counts.items(),
            key=lambda x: (-x[1], x[0])
        )[:k]
    }

    recent_items = []
    seen = set()

    for basket in reversed(baskets):
        for item in basket:
            if item not in seen:
                recent_items.append(item)
                seen.add(item)

            if len(recent_items) == k:
                break

        if len(recent_items) == k:
            break

    recent_items = set(recent_items)

    recency_frequency_disagreement = (
        1 - len(frequent_items & recent_items) / k
    )

    # 7. Long-term preference drift
    # Require enough history for a meaningful early-vs-late comparison.
    if n_baskets >= 4:
        half = n_baskets // 2

        early_items = [
            item
            for basket in baskets[:half]
            for item in basket
        ]

        recent_items_for_drift = [
            item
            for basket in baskets[-half:]
            for item in basket
        ]

        preference_drift = js_divergence(
            Counter(early_items),
            Counter(recent_items_for_drift)
        )
    else:
        preference_drift = np.nan

    # 8. Local regime change
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
        "item_entropy": item_entropy,
        "singleton_item_share": singleton_item_share,
        "adjacent_basket_jaccard": adjacent_basket_jaccard,
        "recency_frequency_disagreement": recency_frequency_disagreement,
        "preference_drift": preference_drift,
        "recent_basket_novelty": recent_basket_novelty,
    }


rows = []

for dataset in DATASETS:
    history_path = ROOT / "datasets" / dataset / "history.json"

    with open(history_path) as f:
        history = json.load(f)

    for user, stored_baskets in history.items():

        # Remove leading/trailing [-1] markers
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