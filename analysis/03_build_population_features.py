from pathlib import Path
from collections import Counter
from itertools import combinations
from datetime import datetime

import json
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


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

TOP_NEIGHBORS = 20
MIN_TRANSITION_SUPPORT = 5


def load_histories(dataset):
    path = ROOT / "datasets" / dataset / "history.json"

    with open(path) as f:
        history = json.load(f)

    # Remove leading/trailing [-1] markers.
    # Treat each basket as a set of items.
    histories = {
        user: [set(basket) for basket in baskets[1:-1]]
        for user, baskets in history.items()
    }

    return histories


def build_item_statistics(histories):
    item_counts = Counter()
    pair_counts = Counter()

    total_baskets = 0

    for baskets in histories.values():
        for basket in baskets:
            total_baskets += 1

            for item in basket:
                item_counts[item] += 1

            for i, j in combinations(sorted(basket), 2):
                pair_counts[(i, j)] += 1

    return item_counts, pair_counts, total_baskets


def build_popularity_percentiles(item_counts):
    df = pd.DataFrame({
        "item": list(item_counts.keys()),
        "count": list(item_counts.values()),
    })

    df["percentile"] = df["count"].rank(pct=True)

    return dict(zip(df["item"], df["percentile"]))


def build_transition_predictability(histories):
    transition_counts = {}
    outgoing_totals = Counter()

    # Number of basket-to-basket transitions in which item i
    # appeared in the origin basket.
    origin_support = Counter()

    for baskets in histories.values():
        for t in range(len(baskets) - 1):
            current = baskets[t]
            nxt = baskets[t + 1]

            for i in current:
                origin_support[i] += 1

                if i not in transition_counts:
                    transition_counts[i] = Counter()

                for j in nxt:
                    transition_counts[i][j] += 1
                    outgoing_totals[i] += 1

    item_predictability = {}

    for i, successors in transition_counts.items():

        # Ignore transition distributions estimated from
        # fewer than 5 observed origin-basket transitions.
        if origin_support[i] < MIN_TRANSITION_SUPPORT:
            continue

        total = outgoing_totals[i]

        probs = np.array([
            count / total
            for count in successors.values()
        ])

        if len(probs) == 1:
            q = 1.0
        else:
            entropy = -np.sum(probs * np.log(probs))
            q = 1 - entropy / np.log(len(probs))

        item_predictability[i] = q

    return item_predictability


def build_user_matrix(histories):
    users = list(histories.keys())

    all_items = sorted({
        item
        for baskets in histories.values()
        for basket in baskets
        for item in basket
    })

    item_to_col = {
        item: idx
        for idx, item in enumerate(all_items)
    }

    rows = []
    cols = []
    values = []

    for row_idx, user in enumerate(users):
        counts = Counter(
            item
            for basket in histories[user]
            for item in basket
        )

        for item, count in counts.items():
            rows.append(row_idx)
            cols.append(item_to_col[item])
            values.append(count)

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(users), len(all_items)),
    )

    return users, matrix


def compute_neighbor_similarity(histories):
    print(f"{datetime.now()}:     building user-item matrix...")

    users, matrix = build_user_matrix(histories)

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])

    print(
        f"{datetime.now()}:     user-item matrix: "
        f"{matrix.shape[0]} users x {matrix.shape[1]} items, "
        f"{matrix.nnz} nonzeros, density={density:.6f}"
    )

    n_neighbors = min(TOP_NEIGHBORS + 1, len(users))

    print(
        f"{datetime.now()}:     fitting NearestNeighbors "
        f"(k={n_neighbors}, cosine/brute)..."
    )

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    model.fit(matrix)

    print(
        f"{datetime.now()}:     querying k-nearest neighbors "
        f"for all users..."
    )

    distances, indices = model.kneighbors(matrix)

    similarities = 1 - distances

    result = {}

    for row_idx, user in enumerate(users):
        # First neighbor is the user themself.
        neighbor_sims = similarities[row_idx][1:]
        result[user] = neighbor_sims.mean()

    print(f"{datetime.now()}:     neighbor similarity done")

    return result


def compute_user_features(
    user,
    baskets,
    item_counts,
    pair_counts,
    popularity_percentiles,
    item_transition_predictability,
    neighbor_similarity,
):
    counts = Counter(
        item
        for basket in baskets
        for item in basket
    )

    # --------------------------------------------------------
    # 1. Popularity profile
    # --------------------------------------------------------

    popularity_profile = np.mean([
        popularity_percentiles[item]
        for item, count in counts.items()
        for _ in range(count)
    ])

    # --------------------------------------------------------
    # 2. Frequency-popularity alignment
    # --------------------------------------------------------

    frequency_popularity_alignment = np.nan

    if len(counts) >= 3:
        local_freq = []
        global_freq = []

        for item, count in counts.items():
            local_freq.append(count)
            global_freq.append(item_counts[item])

        if (
            len(set(local_freq)) > 1
            and len(set(global_freq)) > 1
        ):
            frequency_popularity_alignment = spearmanr(
                local_freq,
                global_freq,
            ).statistic

    # --------------------------------------------------------
    # 3. Co-occurrence support
    #
    # For every pair that appears together in this user's
    # baskets, measure how often that pair occurs in the
    # population. log(1) = 0 for a pair observed only once.
    # --------------------------------------------------------

    pair_supports = []

    for basket in baskets:
        for i, j in combinations(sorted(basket), 2):
            pair_supports.append(
                np.log(pair_counts[(i, j)])
            )

    cooccurrence_support = (
        np.mean(pair_supports)
        if pair_supports
        else np.nan
    )

    # --------------------------------------------------------
    # 4. Transition predictability
    #
    # Average predictability of items in the most recent
    # observed historical basket.
    # --------------------------------------------------------

    latest_basket = baskets[-1]

    q_values = [
        item_transition_predictability[item]
        for item in latest_basket
        if item in item_transition_predictability
    ]

    transition_predictability = (
        np.mean(q_values)
        if q_values
        else np.nan
    )

    return {
        "popularity_profile": popularity_profile,
        "frequency_popularity_alignment":
            frequency_popularity_alignment,
        "neighbor_similarity":
            neighbor_similarity[user],
        "cooccurrence_support":
            cooccurrence_support,
        "transition_predictability":
            transition_predictability,
    }


all_rows = []

output_path = (
    ROOT
    / "analysis"
    / "data"
    / "user_population_features.csv"
)

print(f"Writing to {output_path}")
print(
    f"{datetime.now()}: Starting population features "
    f"for {len(DATASETS)} datasets"
)


for dataset_idx, dataset in enumerate(DATASETS, start=1):

    print(
        f"{datetime.now()}: "
        f"[{dataset_idx}/{len(DATASETS)}] "
        f"Processing {dataset}..."
    )

    print(f"{datetime.now()}:   loading histories...")

    histories = load_histories(dataset)

    n_baskets = sum(
        len(baskets)
        for baskets in histories.values()
    )

    print(
        f"{datetime.now()}:   loaded "
        f"{len(histories)} users, "
        f"{n_baskets} baskets"
    )

    print(
        f"{datetime.now()}:   building item statistics..."
    )

    item_counts, pair_counts, total_baskets = (
        build_item_statistics(histories)
    )

    print(
        f"{datetime.now()}:   "
        f"{len(item_counts)} items, "
        f"{len(pair_counts)} co-occurring pairs, "
        f"{total_baskets} baskets"
    )

    print(
        f"{datetime.now()}:   building popularity percentiles..."
    )

    popularity_percentiles = (
        build_popularity_percentiles(item_counts)
    )

    print(
        f"{datetime.now()}:   building transition statistics..."
    )

    item_transition_predictability = (
        build_transition_predictability(histories)
    )

    print(
        f"{datetime.now()}:   transition predictability "
        f"available for "
        f"{len(item_transition_predictability)} items"
    )

    print(
        f"{datetime.now()}:   computing nearest neighbors..."
    )

    neighbor_similarity = (
        compute_neighbor_similarity(histories)
    )

    print(
        f"{datetime.now()}:   computing per-user features..."
    )

    for user, baskets in tqdm(
        histories.items(),
        total=len(histories),
        desc=f"{dataset} users",
    ):

        features = compute_user_features(
            user=user,
            baskets=baskets,
            item_counts=item_counts,
            pair_counts=pair_counts,
            popularity_percentiles=
                popularity_percentiles,
            item_transition_predictability=
                item_transition_predictability,
            neighbor_similarity=
                neighbor_similarity,
        )

        all_rows.append({
            "dataset": dataset,
            "user": user,
            **features,
        })

    print(
        f"{datetime.now()}:   finished {dataset} "
        f"({len(all_rows)} rows so far)"
    )


print(
    f"{datetime.now()}: Writing feature table..."
)

features = pd.DataFrame(all_rows)

features.to_csv(
    output_path,
    index=False,
)

print(
    f"{datetime.now()}: Feature table shape: "
    f"{features.shape}"
)

print("\nMissing values:")
print(features.isna().sum())

print("\nDataset-level feature means:")
print(
    features.groupby("dataset")
    .mean(numeric_only=True)
    .round(3)
)

print(
    f"{datetime.now()}: Saved to {output_path}"
)