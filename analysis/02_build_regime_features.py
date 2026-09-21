from pathlib import Path
from collections import Counter
from itertools import combinations
from datetime import datetime
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"
DATASETS = ["dunnhumby", "instacart", "retailrocket", "sam", "taobao", "valuedshopper", "tmall", "tafeng"]

TOP_NEIGHBORS = 20
ASSOCIATION_SHRINKAGE = 5.0

DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_histories(dataset):
    with open(ROOT / "datasets" / dataset / "history.json") as f:
        history = json.load(f)
    return {str(user): [set(basket) for basket in baskets[1:-1]] for user, baskets in history.items()}


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else np.nan


def compute_history_features(baskets):
    n_baskets = len(baskets)
    if n_baskets == 0:
        return {
            "n_baskets": 0,
            "avg_basket_size": np.nan,
            "repeat_rate": np.nan,
            "adjacent_basket_jaccard": np.nan,
            "recent_basket_novelty": np.nan,
        }

    basket_sizes = [len(b) for b in baskets]
    all_items = [item for basket in baskets for item in basket]
    n_purchases, n_unique = len(all_items), len(set(all_items))

    adjacent = [jaccard(baskets[t], baskets[t + 1]) for t in range(n_baskets - 1)] if n_baskets >= 2 else []
    adjacent = [x for x in adjacent if not np.isnan(x)]

    if n_baskets >= 2 and baskets[-1]:
        previous_items = set().union(*baskets[:-1])
        recent_novelty = len(baskets[-1] - previous_items) / len(baskets[-1])
    else:
        recent_novelty = np.nan

    return {
        "n_baskets": n_baskets,
        "avg_basket_size": float(np.mean(basket_sizes)),
        "repeat_rate": 1 - n_unique / n_purchases if n_purchases else np.nan,
        "adjacent_basket_jaccard": float(np.mean(adjacent)) if adjacent else np.nan,
        "recent_basket_novelty": recent_novelty,
    }


def iter_transitions(baskets):
    for current, nxt in zip(baskets[:-1], baskets[1:]):
        if current and nxt:
            yield current, nxt


def build_population_statistics(histories):
    item_basket_counts, pair_counts = Counter(), Counter()
    origin_counts, next_counts, transition_pair_counts = Counter(), Counter(), Counter()
    total_baskets = total_transitions = 0

    for baskets in histories.values():
        for basket in baskets:
            if not basket:
                continue

            total_baskets += 1
            item_basket_counts.update(basket)
            pair_counts.update(combinations(sorted(basket), 2))

        for current, nxt in iter_transitions(baskets):
            total_transitions += 1
            origin_counts.update(current)
            next_counts.update(nxt)

            for i in current:
                for j in nxt:
                    if i != j:
                        transition_pair_counts[(i, j)] += 1

    return (
        item_basket_counts,
        pair_counts,
        total_baskets,
        origin_counts,
        next_counts,
        transition_pair_counts,
        total_transitions,
    )


def build_popularity_percentiles(item_basket_counts):
    if not item_basket_counts:
        return {}

    df = pd.DataFrame({"item": list(item_basket_counts), "count": list(item_basket_counts.values())})
    df["percentile"] = df["count"].rank(pct=True, method="average")
    return dict(zip(df["item"], df["percentile"]))


def compute_popularity_profile(baskets, popularity_percentiles):
    counts = Counter(item for basket in baskets for item in basket)
    total = sum(counts.values())
    return float(sum(count * popularity_percentiles[item] for item, count in counts.items()) / total) if total else np.nan


def shrunken_positive_npmi(joint_count, p_i, p_j, p_ij):
    denominator = -np.log(p_ij)

    if p_i <= 0 or p_j <= 0 or p_ij <= 0 or denominator <= 0:
        return 0.0

    npmi = np.log(p_ij / (p_i * p_j)) / denominator
    return (joint_count / (joint_count + ASSOCIATION_SHRINKAGE)) * max(0.0, npmi)


def build_pair_associations(item_basket_counts, pair_counts, total_baskets):
    associations = {}

    if total_baskets == 0:
        return associations

    for pair, pair_count in pair_counts.items():
        i, j = pair
        associations[pair] = shrunken_positive_npmi(
            joint_count=pair_count,
            p_i=item_basket_counts[i] / total_baskets,
            p_j=item_basket_counts[j] / total_baskets,
            p_ij=pair_count / total_baskets,
        )

    return associations


def compute_relational_strength(baskets, pair_associations):
    valid_baskets = [basket for basket in baskets if basket]

    if not valid_baskets:
        return np.nan

    scores = []
    for basket in valid_baskets:
        if len(basket) < 2:
            scores.append(0.0)
            continue

        pair_scores = [pair_associations.get(pair, 0.0) for pair in combinations(sorted(basket), 2)]
        scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)

    return float(np.mean(scores))


def build_transition_associations(
    origin_counts,
    next_counts,
    transition_pair_counts,
    total_transitions,
):
    """
    Directed presence NPMI for i in B_t, j in B_{t+1}, i != j.

    P_→(i,j) = c_→(i,j) / T
    P_origin(i) = c_origin(i) / T
    P_next(j) = c_next(j) / T

    These are binary presence rates, so they need not sum to 1.
    Diagonal pairs are omitted: they are never used in the user score.
    """
    associations = {}

    if total_transitions == 0:
        return associations

    for pair, pair_count in transition_pair_counts.items():
        i, j = pair
        associations[pair] = shrunken_positive_npmi(
            joint_count=pair_count,
            p_i=origin_counts[i] / total_transitions,
            p_j=next_counts[j] / total_transitions,
            p_ij=pair_count / total_transitions,
        )

    return associations


def compute_transition_strength(baskets, transition_associations):
    scores = []

    for current, nxt in iter_transitions(baskets):
        pair_scores = [
            transition_associations.get((i, j), 0.0)
            for i in current
            for j in nxt
            if i != j
        ]
        scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)

    if not scores:
        return np.nan

    return float(np.mean(scores))


def build_idf_weighted_user_matrix(histories):
    users = list(histories)
    all_items = sorted({item for baskets in histories.values() for basket in baskets for item in basket})
    item_to_col = {item: idx for idx, item in enumerate(all_items)}

    rows, cols, values = [], [], []

    for row_idx, user in enumerate(users):
        counts = Counter(item for basket in histories[user] for item in basket)

        for item, count in counts.items():
            rows.append(row_idx)
            cols.append(item_to_col[item])
            values.append(count)

    matrix = csr_matrix((values, (rows, cols)), shape=(len(users), len(all_items)), dtype=float)

    if matrix.shape[1] == 0:
        return users, matrix

    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    idf = np.log((len(users) + 1) / (document_frequency + 1)) + 1.0
    matrix.data = np.log1p(matrix.data)

    return users, matrix.multiply(idf).tocsr()


def compute_neighbor_similarity(histories):
    users, matrix = build_idf_weighted_user_matrix(histories)

    if len(users) < 2 or matrix.shape[1] == 0:
        return {user: np.nan for user in users}

    n_neighbors = min(TOP_NEIGHBORS + 1, len(users))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(matrix)

    distances, _ = model.kneighbors(matrix)
    similarities = 1.0 - distances

    return {
        user: float(np.mean(similarities[row_idx][1:])) if n_neighbors > 1 else np.nan
        for row_idx, user in enumerate(users)
    }


rows = []
output_path = DATA_DIR / "regime_features.csv"

for dataset_idx, dataset in enumerate(DATASETS, start=1):
    print(f"\n{datetime.now()}: [{dataset_idx}/{len(DATASETS)}] {dataset}")
    histories = load_histories(dataset)

    (
        item_basket_counts,
        pair_counts,
        total_baskets,
        origin_counts,
        next_counts,
        transition_pair_counts,
        total_transitions,
    ) = build_population_statistics(histories)

    print(
        f"{datetime.now()}: {len(histories)} users, "
        f"{total_baskets} baskets, {len(pair_counts)} undirected pairs, "
        f"{total_transitions} transitions, {len(transition_pair_counts)} directed pairs"
    )

    popularity_percentiles = build_popularity_percentiles(item_basket_counts)
    pair_associations = build_pair_associations(
        item_basket_counts,
        pair_counts,
        total_baskets,
    )
    transition_associations = build_transition_associations(
        origin_counts,
        next_counts,
        transition_pair_counts,
        total_transitions,
    )

    print(f"{datetime.now()}: computing neighbor similarity...")
    neighbor_similarity = compute_neighbor_similarity(histories)

    print(f"{datetime.now()}: computing user features...")

    for user, baskets in tqdm(histories.items(), total=len(histories), desc=dataset):
        row = {"dataset": dataset, "user": user, **compute_history_features(baskets)}
        row["popularity_profile"] = compute_popularity_profile(baskets, popularity_percentiles)
        row["neighbor_similarity"] = neighbor_similarity[user]
        row["transition_strength"] = compute_transition_strength(baskets, transition_associations)
        row["relational_strength"] = compute_relational_strength(baskets, pair_associations)
        rows.append(row)

features = pd.DataFrame(rows)

assert not features.duplicated(["dataset", "user"]).any()

features.to_csv(output_path, index=False)

print("\nRegime feature table shape:", features.shape)
print("\nUsers by dataset:")
print(features.groupby("dataset").size())
print("\nMissing values:")
print(features.isna().sum())
print(f"\nSaved to {output_path}")
