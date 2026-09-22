from pathlib import Path
from collections import Counter
from itertools import combinations
from datetime import datetime
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

DATASETS = ["dunnhumby", "instacart", "sam", "tafeng", "valuedshopper"]
METHODS = ["recency", "frequency", "tifuknn", "fpmc", "triple2vec", "dnntsp", "cbp", "diffrec"]
KEYSETS = [0, 1, 2]

TOP_K = 5
TOP_NEIGHBORS = 20
ASSOCIATION_SHRINKAGE = 5.0
N_BINS = 5
EPS = 1e-12

TARGET_FEATURES = [
    "target_basket_size",
    "target_repeat_fraction",
    "target_last_jaccard",
    "target_novelty_fraction",
    "target_popularity_profile",
    "target_neighbor_support",
    "target_transition_strength",
    "target_relational_strength",
]

ITEM_FEATURES = [
    "personal_frequency",
    "personal_recency_score",
    "global_popularity",
    "neighbor_support",
    "transition_support",
    "relational_support",
]


def normalize_basket(basket):
    return {str(item) for item in basket}


def load_histories(dataset):
    # Match Analysis 02: observable history is history.json[1:-1].
    with open(ROOT / "datasets" / dataset / "history.json") as f:
        raw = json.load(f)
    return {
        str(user): [normalize_basket(b) for b in baskets[1:-1]]
        for user, baskets in raw.items()
    }


def load_targets(dataset):
    # Match evaluator: future.json[user][1] is the test basket.
    with open(ROOT / "datasets" / dataset / "future.json") as f:
        raw = json.load(f)
    return {
        str(user): normalize_basket(future[1])
        for user, future in raw.items()
        if len(future) > 1
    }


def load_test_users(dataset, keyset):
    with open(ROOT / "datasets" / dataset / f"keyset_{keyset}.json") as f:
        split = json.load(f)
    return set(map(str, split["test"]))


def load_predictions(dataset, method, keyset):
    with open(ROOT / "predictions" / dataset / method / f"keyset{keyset}.json") as f:
        raw = json.load(f)
    return {
        str(user): [str(item) for item in preds]
        for user, preds in raw.items()
    }


def iter_transitions(baskets):
    for current, nxt in zip(baskets[:-1], baskets[1:]):
        if current and nxt:
            yield current, nxt


def shrunken_positive_npmi(joint_count, p_i, p_j, p_ij):
    denominator = -np.log(p_ij)
    if p_i <= 0 or p_j <= 0 or p_ij <= 0 or denominator <= 0:
        return 0.0
    npmi = np.log(p_ij / (p_i * p_j)) / denominator
    return (joint_count / (joint_count + ASSOCIATION_SHRINKAGE)) * max(0.0, npmi)


def build_population_statistics(histories):
    item_basket_counts = Counter()
    pair_counts = Counter()
    origin_counts = Counter()
    next_counts = Counter()
    transition_pair_counts = Counter()
    total_baskets = 0
    total_transitions = 0

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

    if item_basket_counts:
        pop = pd.Series(item_basket_counts, dtype=float)
        popularity_percentiles = pop.rank(pct=True, method="average").to_dict()
    else:
        popularity_percentiles = {}

    pair_assoc = {}
    if total_baskets > 0:
        for (i, j), count in pair_counts.items():
            pair_assoc[(i, j)] = shrunken_positive_npmi(
                count,
                item_basket_counts[i] / total_baskets,
                item_basket_counts[j] / total_baskets,
                count / total_baskets,
            )

    transition_assoc = {}
    if total_transitions > 0:
        for (i, j), count in transition_pair_counts.items():
            transition_assoc[(i, j)] = shrunken_positive_npmi(
                count,
                origin_counts[i] / total_transitions,
                next_counts[j] / total_transitions,
                count / total_transitions,
            )

    return popularity_percentiles, pair_assoc, transition_assoc


def build_neighbor_map(histories):
    users = list(histories)
    all_items = sorted({
        item
        for baskets in histories.values()
        for basket in baskets
        for item in basket
    })

    if len(users) < 2 or not all_items:
        return {user: [] for user in users}

    item_to_col = {item: idx for idx, item in enumerate(all_items)}
    rows, cols, vals = [], [], []

    for row_idx, user in enumerate(users):
        counts = Counter(item for basket in histories[user] for item in basket)
        for item, count in counts.items():
            rows.append(row_idx)
            cols.append(item_to_col[item])
            vals.append(count)

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(users), len(all_items)),
        dtype=float,
    )

    df = np.asarray((matrix > 0).sum(axis=0)).ravel()
    idf = np.log((len(users) + 1) / (df + 1)) + 1.0
    matrix.data = np.log1p(matrix.data)
    matrix = matrix.multiply(idf).tocsr()

    n_neighbors = min(TOP_NEIGHBORS + 1, len(users))
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(matrix)
    _, indices = model.kneighbors(matrix)

    out = {}
    for row_idx, user in enumerate(users):
        out[user] = [
            users[j]
            for j in indices[row_idx]
            if users[j] != user
        ][:TOP_NEIGHBORS]
    return out


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else np.nan


def pair_value(i, j, pair_assoc):
    if i == j:
        return 0.0
    return pair_assoc.get(tuple(sorted((i, j))), 0.0)


def personal_frequency(item, baskets):
    return np.mean([item in b for b in baskets]) if baskets else np.nan


def personal_recency_score(item, baskets):
    for lag, basket in enumerate(reversed(baskets)):
        if item in basket:
            return 1.0 / (lag + 1.0)
    return 0.0


def neighbor_support(item, neighbors, history_item_sets):
    if not neighbors:
        return np.nan
    return float(np.mean([
        item in history_item_sets.get(n, set())
        for n in neighbors
    ]))


def transition_support(item, last_basket, transition_assoc):
    if not last_basket:
        return np.nan
    vals = [
        transition_assoc.get((i, item), 0.0)
        for i in last_basket
        if i != item
    ]
    return float(np.mean(vals)) if vals else 0.0


def relational_support(item, target, pair_assoc):
    others = [j for j in target if j != item]
    if not others:
        return 0.0
    return float(np.mean([
        pair_value(item, j, pair_assoc)
        for j in others
    ]))


def build_target_tables(
    dataset,
    histories,
    targets,
    popularity_percentiles,
    pair_assoc,
    transition_assoc,
    neighbor_map,
    analysis_users,
):
    history_item_sets = {
        user: set().union(*baskets) if baskets else set()
        for user, baskets in histories.items()
    }

    user_rows = []
    item_rows = []

    for user in analysis_users:
        baskets = histories.get(user, [])
        target = targets.get(user, set())
        if not target:
            continue

        history_items = history_item_sets.get(user, set())
        last_basket = baskets[-1] if baskets else set()
        neighbors = neighbor_map.get(user, [])

        local_item_rows = []
        for item in target:
            is_repeat = int(item in history_items)
            in_last = int(item in last_basket)

            row = {
                "dataset": dataset,
                "user": user,
                "item": item,
                "is_repeat": is_repeat,
                "in_last_basket": in_last,
                "target_class": (
                    "last_basket_repeat"
                    if in_last
                    else "older_repeat"
                    if is_repeat
                    else "novel"
                ),
                "personal_frequency": personal_frequency(item, baskets),
                "personal_recency_score": personal_recency_score(item, baskets),
                "global_popularity": popularity_percentiles.get(item, 0.0),
                "neighbor_support": neighbor_support(
                    item, neighbors, history_item_sets
                ),
                "transition_support": transition_support(
                    item, last_basket, transition_assoc
                ),
                "relational_support": relational_support(
                    item, target, pair_assoc
                ),
            }
            item_rows.append(row)
            local_item_rows.append(row)

        item_df = pd.DataFrame(local_item_rows)

        user_rows.append({
            "dataset": dataset,
            "user": user,
            "history_n_baskets": len(baskets),
            "target_basket_size": len(target),
            "target_repeat_fraction": item_df["is_repeat"].mean(),
            "target_last_jaccard": (
                jaccard(last_basket, target) if last_basket else np.nan
            ),
            "target_novelty_fraction": 1.0 - item_df["is_repeat"].mean(),
            "target_popularity_profile": item_df["global_popularity"].mean(),
            "target_neighbor_support": item_df["neighbor_support"].mean(),
            "target_transition_strength": item_df["transition_support"].mean(),
            "target_relational_strength": item_df["relational_support"].mean(),
            # Extra bridge mechanisms for FPMC x history depth.
            "target_personal_frequency": item_df["personal_frequency"].mean(),
            "target_personal_recency_score": item_df["personal_recency_score"].mean(),
        })

    return pd.DataFrame(user_rows), pd.DataFrame(item_rows)


def safe_spearman(x, y):
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return x.corr(y, method="spearman")


def add_dataset_bins(df, features):
    out = df.copy()
    for feature in features:
        out[f"{feature}_bin"] = np.nan
        for dataset, idx in out.groupby("dataset").groups.items():
            x = out.loc[idx, feature]
            valid = x.notna()
            if valid.sum() == 0:
                continue
            pct = x.loc[valid].rank(method="average", pct=True)
            out.loc[pct.index, f"{feature}_bin"] = (
                np.ceil(pct * N_BINS).clip(1, N_BINS).astype(int)
            )
    return out


def summarize_target_method_effects(performance, target_features):
    tf = add_dataset_bins(target_features, TARGET_FEATURES)
    analysis = performance.merge(
        tf[
            ["dataset", "user"]
            + TARGET_FEATURES
            + [f"{f}_bin" for f in TARGET_FEATURES]
        ],
        on=["dataset", "user"],
        how="inner",
        validate="many_to_one",
    )

    analysis["user_mean_ndcg"] = analysis[METHODS].mean(axis=1)
    for method in METHODS:
        analysis[f"{method}_centered"] = (
            analysis[method] - analysis["user_mean_ndcg"]
        )

    rows = []
    for (dataset, keyset), df in analysis.groupby(["dataset", "keyset"]):
        for feature in TARGET_FEATURES:
            bin_col = f"{feature}_bin"

            for method in METHODS:
                rho_raw = safe_spearman(df[feature], df[method])
                rho_centered = safe_spearman(
                    df[feature], df[f"{method}_centered"]
                )

                valid = df[feature].notna() & df[bin_col].notna()
                binned = df.loc[valid]
                means = (
                    binned
                    .groupby(bin_col)
                    .agg(
                        raw=(method, "mean"),
                        centered=(f"{method}_centered", "mean"),
                    )
                    .sort_index()
                )

                if len(means) >= 2:
                    raw_effect = means.iloc[-1]["raw"] - means.iloc[0]["raw"]
                    centered_effect = (
                        means.iloc[-1]["centered"]
                        - means.iloc[0]["centered"]
                    )
                else:
                    raw_effect = np.nan
                    centered_effect = np.nan

                rows.append({
                    "dataset": dataset,
                    "keyset": keyset,
                    "feature": feature,
                    "method": method,
                    "n_users": int(valid.sum()),
                    "rho_raw": rho_raw,
                    "rho_centered": rho_centered,
                    "raw_low_to_high": raw_effect,
                    "centered_low_to_high": centered_effect,
                })

    effects = pd.DataFrame(rows)

    def sign_label(values):
        values = pd.Series(values).dropna()
        if len(values) < 3:
            return "missing"
        if (values > 0).all():
            return "positive"
        if (values < 0).all():
            return "negative"
        return "mixed"

    summary = (
        effects
        .groupby(["dataset", "feature", "method"], as_index=False)
        .agg(
            n_users=("n_users", "mean"),
            rho_raw_mean=("rho_raw", "mean"),
            rho_raw_sd=("rho_raw", "std"),
            rho_centered_mean=("rho_centered", "mean"),
            rho_centered_sd=("rho_centered", "std"),
            raw_low_to_high_mean=("raw_low_to_high", "mean"),
            raw_low_to_high_sd=("raw_low_to_high", "std"),
            centered_low_to_high_mean=("centered_low_to_high", "mean"),
            centered_low_to_high_sd=("centered_low_to_high", "std"),
            rho_raw_sign=("rho_raw", sign_label),
            rho_centered_sign=("rho_centered", sign_label),
            centered_effect_sign=("centered_low_to_high", sign_label),
        )
    )
    return summary


def ndcg_item_contribution(rank, target_size):
    # Mirrors evaluator: nDCG only uses first min(k, target_size) slots.
    n = min(TOP_K, target_size)
    if rank is None or rank > n or n == 0:
        return 0.0
    idcg = sum(1.0 / np.log(i + 2) for i in range(n))
    return (1.0 / np.log(rank + 1)) / idcg


def summarize_item_mechanisms(dataset, item_table):
    item_table = add_dataset_bins(item_table, ITEM_FEATURES)
    rows = []

    for keyset in KEYSETS:
        test_users = load_test_users(dataset, keyset)
        base = item_table[item_table["user"].isin(test_users)].copy()
        if base.empty:
            continue

        target_sizes = base.groupby("user").size().to_dict()

        for method in METHODS:
            preds = load_predictions(dataset, method, keyset)
            temp = base.copy()

            ranks = []
            for user, item in zip(temp["user"], temp["item"]):
                top = preds.get(user, [])[:TOP_K]
                try:
                    ranks.append(top.index(item) + 1)
                except ValueError:
                    ranks.append(np.nan)

            temp["rank_at5"] = ranks
            temp["hit_at5"] = temp["rank_at5"].notna().astype(int)
            temp["ndcg_contribution"] = [
                ndcg_item_contribution(
                    int(rank) if pd.notna(rank) else None,
                    target_sizes[user],
                )
                for user, rank in zip(temp["user"], temp["rank_at5"])
            ]

            for cls, g in temp.groupby("target_class"):
                rows.append({
                    "dataset": dataset,
                    "keyset": keyset,
                    "method": method,
                    "analysis_type": "target_class",
                    "mechanism": "target_class",
                    "category": cls,
                    "n_items": len(g),
                    "hit_rate_at5": g["hit_at5"].mean(),
                    "mean_ndcg_contribution": g["ndcg_contribution"].mean(),
                    "rho_hit": np.nan,
                    "rho_ndcg": np.nan,
                    "low_to_high_hit": np.nan,
                    "low_to_high_ndcg": np.nan,
                })

            for feature in ITEM_FEATURES:
                g = temp[temp[feature].notna()].copy()
                bin_col = f"{feature}_bin"

                rho_hit = safe_spearman(g[feature], g["hit_at5"])
                rho_ndcg = safe_spearman(
                    g[feature], g["ndcg_contribution"]
                )

                b = (
                    g[g[bin_col].notna()]
                    .groupby(bin_col)
                    .agg(
                        hit=("hit_at5", "mean"),
                        ndcg=("ndcg_contribution", "mean"),
                    )
                    .sort_index()
                )

                if len(b) >= 2:
                    hit_effect = b.iloc[-1]["hit"] - b.iloc[0]["hit"]
                    ndcg_effect = b.iloc[-1]["ndcg"] - b.iloc[0]["ndcg"]
                else:
                    hit_effect = np.nan
                    ndcg_effect = np.nan

                rows.append({
                    "dataset": dataset,
                    "keyset": keyset,
                    "method": method,
                    "analysis_type": "continuous",
                    "mechanism": feature,
                    "category": "",
                    "n_items": len(g),
                    "hit_rate_at5": g["hit_at5"].mean(),
                    "mean_ndcg_contribution": g["ndcg_contribution"].mean(),
                    "rho_hit": rho_hit,
                    "rho_ndcg": rho_ndcg,
                    "low_to_high_hit": hit_effect,
                    "low_to_high_ndcg": ndcg_effect,
                })

    return pd.DataFrame(rows)


def history_depth_summary(target_features):
    rows = []

    for dataset, df in target_features.groupby("dataset"):
        pct = df["history_n_baskets"].rank(method="average", pct=True)
        temp = df.copy()
        temp["history_depth_bin"] = (
            np.ceil(pct * N_BINS).clip(1, N_BINS).astype(int)
        )

        for bin_id, g in temp.groupby("history_depth_bin"):
            rows.append({
                "dataset": dataset,
                "history_depth_bin": int(bin_id),
                "n_users": len(g),
                "history_n_baskets_mean": g["history_n_baskets"].mean(),
                "target_repeat_fraction": g["target_repeat_fraction"].mean(),
                "target_last_jaccard": g["target_last_jaccard"].mean(),
                "target_personal_frequency": g["target_personal_frequency"].mean(),
                "target_personal_recency_score": g["target_personal_recency_score"].mean(),
                "target_transition_strength": g["target_transition_strength"].mean(),
                "target_relational_strength": g["target_relational_strength"].mean(),
                "target_popularity_profile": g["target_popularity_profile"].mean(),
                "target_neighbor_support": g["target_neighbor_support"].mean(),
            })

    return pd.DataFrame(rows)


print(f"{datetime.now()}: loading performance", flush=True)

performance = pd.read_csv(
    DATA_DIR / "performance.csv",
    dtype={"user": str},
)
performance = performance[
    performance["dataset"].isin(DATASETS)
].copy()

target_parts = []
item_tables = {}

for dataset in DATASETS:
    print(f"\n{datetime.now()}: {dataset}: loading source data", flush=True)
    histories = load_histories(dataset)
    targets = load_targets(dataset)
    analysis_users = set(
        performance.loc[performance["dataset"] == dataset, "user"].unique()
    )

    print(f"{datetime.now()}: {dataset}: population statistics", flush=True)
    popularity, pair_assoc, transition_assoc = build_population_statistics(
        histories
    )

    print(f"{datetime.now()}: {dataset}: neighbor map", flush=True)
    neighbor_map = build_neighbor_map(histories)

    print(f"{datetime.now()}: {dataset}: target features", flush=True)
    user_table, item_table = build_target_tables(
        dataset,
        histories,
        targets,
        popularity,
        pair_assoc,
        transition_assoc,
        neighbor_map,
        analysis_users,
    )

    target_parts.append(user_table)
    item_tables[dataset] = item_table

target_features = pd.concat(target_parts, ignore_index=True)
assert not target_features.duplicated(["dataset", "user"]).any()

print(f"\n{datetime.now()}: target-level method effects", flush=True)
target_effects = summarize_target_method_effects(
    performance, target_features
)

print(f"{datetime.now()}: item-level mechanisms from saved predictions", flush=True)
item_summaries = []
for dataset in DATASETS:
    print(f"{datetime.now()}: {dataset}", flush=True)
    item_summaries.append(
        summarize_item_mechanisms(dataset, item_tables[dataset])
    )

item_mechanisms = pd.concat(item_summaries, ignore_index=True)
depth_mechanisms = history_depth_summary(target_features)

target_features.to_csv(
    DATA_DIR / "target_mechanism_features.csv",
    index=False,
)
target_effects.to_csv(
    DATA_DIR / "target_method_effects.csv",
    index=False,
)
item_mechanisms.to_csv(
    DATA_DIR / "target_item_mechanisms.csv",
    index=False,
)
depth_mechanisms.to_csv(
    DATA_DIR / "history_depth_mechanisms.csv",
    index=False,
)

print("\nSaved:")
print(DATA_DIR / "target_mechanism_features.csv")
print(DATA_DIR / "target_method_effects.csv")
print(DATA_DIR / "target_item_mechanisms.csv")
print(DATA_DIR / "history_depth_mechanisms.csv")

print("\nInterpretation notes:")
print("  target_repeat_fraction and target_novelty_fraction are exact complements.")
print("  last_basket_repeat / older_repeat / novel is the key item-level decomposition.")
print("  population statistics are built from observed history only.")
print("  future labels are retrospective mechanism labels, not prediction-time regime features.")
print("  saved predictions are reused; this script performs no training or inference.")
