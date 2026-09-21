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
    "fpmc",
    "triple2vec",
    "sets2sets",
    "dnntsp",
    "cbp",
    "diffrec",
]

EPS = 1e-12


# ------------------------------------------------------------
# PART 1
# Keyset stability of feature -> model advantage effects
# ------------------------------------------------------------

summary = pd.read_csv(
    DATA_DIR
    / "feature_advantage_correlation_summary.csv"
)


RHO_COLUMNS = [
    "rho_keyset0",
    "rho_keyset1",
    "rho_keyset2",
]


# Number of usable keysets
summary["n_valid_keysets"] = (
    summary[RHO_COLUMNS]
    .notna()
    .sum(axis=1)
)


# Standard deviation across keysets
summary["rho_sd"] = (
    summary[RHO_COLUMNS]
    .std(
        axis=1,
        ddof=0,
    )
)


# Largest deviation of one keyset
# from the mean correlation
summary["max_deviation_from_mean"] = (
    summary[RHO_COLUMNS]
    .sub(
        summary["mean_rho"],
        axis=0,
    )
    .abs()
    .max(axis=1)
)


# Replication classification
def replication_status(row):

    values = row[RHO_COLUMNS]

    if values.isna().any():
        return "missing"

    if (values > 0).all():
        return "positive"

    if (values < 0).all():
        return "negative"

    return "mixed"


summary["replication_status"] = (
    summary.apply(
        replication_status,
        axis=1,
    )
)


summary["replicated"] = (
    summary["replication_status"]
    .isin([
        "positive",
        "negative",
    ])
)


summary.to_csv(
    DATA_DIR
    / "keyset_stability.csv",
    index=False,
)


# ------------------------------------------------------------
# Replicated effects only
#
# NO arbitrary effect-size threshold.
# Just require the same direction in all 3 keysets.
# ------------------------------------------------------------

replicated = (
    summary[
        summary["replicated"]
        & (summary["n_valid_keysets"] == 3)
    ]
    .copy()
)


replicated = replicated.sort_values(
    [
        "dataset",
        "abs_mean_rho",
    ],
    ascending=[
        True,
        False,
    ],
)


replicated.to_csv(
    DATA_DIR
    / "replicated_feature_advantages.csv",
    index=False,
)


# ------------------------------------------------------------
# PART 2
# Pairwise decisiveness and conditional win rates
#
# This tells us whether apparent model heterogeneity
# reflects frequent disagreement or only rare differences.
# ------------------------------------------------------------

master = pd.read_csv(
    DATA_DIR / "master_analysis.csv",
    dtype={"user": str},
)


pair_rows = []

for (dataset, keyset), df in master.groupby(
    ["dataset", "keyset"]
):

    for method_a, method_b in combinations(
        METHODS,
        2,
    ):

        delta = (
            df[method_a]
            - df[method_b]
        )

        a_wins = (
            delta > EPS
        )

        b_wins = (
            delta < -EPS
        )

        ties = (
            delta.abs() <= EPS
        )

        n = len(delta)

        n_a = a_wins.sum()
        n_b = b_wins.sum()
        n_ties = ties.sum()

        n_decisive = (
            n_a + n_b
        )

        decisive_rate = (
            n_decisive / n
            if n > 0
            else np.nan
        )

        conditional_a_win_rate = (
            n_a / n_decisive
            if n_decisive > 0
            else np.nan
        )

        conditional_b_win_rate = (
            n_b / n_decisive
            if n_decisive > 0
            else np.nan
        )

        pair_rows.append({
            "dataset": dataset,
            "keyset": keyset,
            "method_a": method_a,
            "method_b": method_b,
            "n_users": n,
            "n_decisive": n_decisive,
            "a_win_rate": n_a / n,
            "b_win_rate": n_b / n,
            "tie_rate": n_ties / n,
            "decisive_rate": decisive_rate,
            "conditional_a_win_rate":
                conditional_a_win_rate,
            "conditional_b_win_rate":
                conditional_b_win_rate,
            "mean_delta": delta.mean(),
        })


pairwise = pd.DataFrame(pair_rows)

pairwise.to_csv(
    DATA_DIR
    / "pairwise_decisiveness_by_keyset.csv",
    index=False,
)


# ------------------------------------------------------------
# Summarize pairwise decisiveness across keysets
# ------------------------------------------------------------

pair_summary = (
    pairwise
    .groupby(
        [
            "dataset",
            "method_a",
            "method_b",
        ],
        as_index=False,
    )
    .agg(
        mean_delta=("mean_delta", "mean"),
        mean_decisive_rate=(
            "decisive_rate",
            "mean",
        ),
        mean_conditional_a_win_rate=(
            "conditional_a_win_rate",
            "mean",
        ),
        mean_tie_rate=(
            "tie_rate",
            "mean",
        ),
    )
)


pair_summary.to_csv(
    DATA_DIR
    / "pairwise_decisiveness_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print(
    "\nReplicated feature-advantage "
    "effects by dataset:"
)

for dataset in summary["dataset"].unique():

    temp = (
        replicated[
            replicated["dataset"] == dataset
        ]
        .head(15)
    )

    print(f"\n--- {dataset} ---")

    print(
        temp[
            [
                "method_a",
                "method_b",
                "feature",
                "rho_keyset0",
                "rho_keyset1",
                "rho_keyset2",
                "mean_rho",
                "rho_sd",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


print(
    "\nNumber of replicated effects "
    "by dataset:"
)

print(
    replicated
    .groupby("dataset")
    .size()
)


print(
    "\nPairwise comparisons with "
    "highest disagreement rates:"
)

for dataset in pair_summary["dataset"].unique():

    temp = (
        pair_summary[
            pair_summary["dataset"] == dataset
        ]
        .sort_values(
            "mean_decisive_rate",
            ascending=False,
        )
        .head(10)
    )

    print(f"\n--- {dataset} ---")

    print(
        temp[
            [
                "method_a",
                "method_b",
                "mean_delta",
                "mean_decisive_rate",
                "mean_conditional_a_win_rate",
                "mean_tie_rate",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


print("\nSaved:")
print(
    DATA_DIR
    / "keyset_stability.csv"
)
print(
    DATA_DIR
    / "replicated_feature_advantages.csv"
)
print(
    DATA_DIR
    / "pairwise_decisiveness_by_keyset.csv"
)
print(
    DATA_DIR
    / "pairwise_decisiveness_summary.csv"
)