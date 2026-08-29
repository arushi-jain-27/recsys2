from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

METHODS = [
    "recency",
    "frequency",
    "racf",
    "tifuknn",
    "triple2vec",
    "sets2sets",
    "dnntsp",
]

EPS = 1e-12


master = pd.read_csv(
    DATA_DIR / "master_analysis.csv",
    dtype={"user": str},
)


# ------------------------------------------------------------
# 1. Aggregate performance by dataset/keyset
# ------------------------------------------------------------

aggregate = (
    master
    .groupby(["dataset", "keyset"])[METHODS]
    .mean()
    .reset_index()
)

aggregate.to_csv(
    DATA_DIR / "performance_by_keyset.csv",
    index=False,
)


# ------------------------------------------------------------
# 2. User-level winner shares
#
# We exclude all-zero users from winner shares because no
# method successfully predicted anything for them.
#
# Exact ties receive fractional winner credit.
# ------------------------------------------------------------

winner_rows = []

for (dataset, keyset), df in master.groupby(
    ["dataset", "keyset"]
):

    credits = {
        method: 0.0
        for method in METHODS
    }

    n_nonzero = 0
    n_unique_winner = 0
    n_tied_winner = 0
    n_all_zero = 0

    for _, row in df.iterrows():

        scores = row[METHODS].to_numpy(dtype=float)
        best = scores.max()

        if best <= EPS:
            n_all_zero += 1
            continue

        n_nonzero += 1

        winners = [
            METHODS[i]
            for i, score in enumerate(scores)
            if abs(score - best) <= EPS
        ]

        if len(winners) == 1:
            n_unique_winner += 1
        else:
            n_tied_winner += 1

        credit = 1.0 / len(winners)

        for method in winners:
            credits[method] += credit

    row = {
        "dataset": dataset,
        "keyset": keyset,
        "n_users": len(df),
        "all_zero_rate": n_all_zero / len(df),
        "unique_winner_rate":
            n_unique_winner / n_nonzero
            if n_nonzero else np.nan,
        "tied_winner_rate":
            n_tied_winner / n_nonzero
            if n_nonzero else np.nan,
    }

    for method in METHODS:
        row[f"{method}_winner_share"] = (
            credits[method] / n_nonzero
            if n_nonzero else np.nan
        )

    winner_rows.append(row)


winner_shares = pd.DataFrame(winner_rows)

winner_shares.to_csv(
    DATA_DIR / "winner_shares.csv",
    index=False,
)


# ------------------------------------------------------------
# 3. Pairwise model advantages
# ------------------------------------------------------------

pairwise_rows = []

for (dataset, keyset), df in master.groupby(
    ["dataset", "keyset"]
):

    for method_a, method_b in combinations(METHODS, 2):

        delta = (
            df[method_a] - df[method_b]
        )

        pairwise_rows.append({
            "dataset": dataset,
            "keyset": keyset,
            "method_a": method_a,
            "method_b": method_b,
            "mean_delta": delta.mean(),
            "median_delta": delta.median(),
            "p10_delta": delta.quantile(0.10),
            "p90_delta": delta.quantile(0.90),
            "a_better_share": (delta > EPS).mean(),
            "b_better_share": (delta < -EPS).mean(),
            "tie_share":
                (delta.abs() <= EPS).mean(),
        })


pairwise = pd.DataFrame(pairwise_rows)

pairwise["two_sided_share"] = pairwise[
    ["a_better_share", "b_better_share"]
].min(axis=1)

pairwise.to_csv(
    DATA_DIR / "pairwise_advantage_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# 4. Oracle headroom
#
# This is descriptive only:
# how much better could we do if we somehow knew the
# best method for every user?
# ------------------------------------------------------------

oracle_rows = []

for (dataset, keyset), df in master.groupby(
    ["dataset", "keyset"]
):

    method_means = df[METHODS].mean()

    best_method = method_means.idxmax()
    best_mean = method_means.max()

    oracle_mean = (
        df[METHODS]
        .max(axis=1)
        .mean()
    )

    oracle_rows.append({
        "dataset": dataset,
        "keyset": keyset,
        "best_method": best_method,
        "best_method_ndcg": best_mean,
        "oracle_ndcg": oracle_mean,
        "oracle_headroom":
            oracle_mean - best_mean,
    })


oracle = pd.DataFrame(oracle_rows)

oracle.to_csv(
    DATA_DIR / "oracle_headroom.csv",
    index=False,
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print("\nMean performance across keysets:")
print(
    aggregate
    .groupby("dataset")[METHODS]
    .mean()
    .round(4)
)


print("\nWinner shares across keysets:")

winner_columns = [
    f"{method}_winner_share"
    for method in METHODS
]

print(
    winner_shares
    .groupby("dataset")[winner_columns]
    .mean()
    .round(3)
)


print("\nAll-zero prediction rate:")

print(
    winner_shares
    .groupby("dataset")["all_zero_rate"]
    .mean()
    .round(3)
)


print("\nOracle headroom:")

print(
    oracle
    .groupby("dataset")[
        [
            "best_method_ndcg",
            "oracle_ndcg",
            "oracle_headroom",
        ]
    ]
    .mean()
    .round(4)
)


print("\nMost heterogeneous pairwise comparisons:")

for dataset in master["dataset"].unique():

    temp = (
        pairwise[
            pairwise["dataset"] == dataset
        ]
        .groupby(
            ["method_a", "method_b"],
            as_index=False,
        )[
            [
                "mean_delta",
                "a_better_share",
                "b_better_share",
                "tie_share",
                "two_sided_share",
            ]
        ]
        .mean()
        .sort_values(
            "two_sided_share",
            ascending=False,
        )
        .head(5)
    )

    print(f"\n--- {dataset} ---")

    print(
        temp[
            [
                "method_a",
                "method_b",
                "mean_delta",
                "a_better_share",
                "b_better_share",
                "tie_share",
            ]
        ].round(3).to_string(index=False)
    )


print("\nSaved:")
print(DATA_DIR / "performance_by_keyset.csv")
print(DATA_DIR / "winner_shares.csv")
print(DATA_DIR / "pairwise_advantage_summary.csv")
print(DATA_DIR / "oracle_headroom.csv")