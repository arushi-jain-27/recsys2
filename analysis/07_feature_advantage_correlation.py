from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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

FEATURES = [
    "n_baskets",
    "avg_basket_size",
    "repeat_rate",
    "adjacent_basket_jaccard",
    "recency_frequency_disagreement",
    "recent_basket_novelty",
    "popularity_profile",
    "neighbor_similarity",
    "cooccurrence_support",
    "transition_predictability",
]


master = pd.read_csv(
    DATA_DIR / "master_analysis.csv",
    dtype={"user": str},
)


# ------------------------------------------------------------
# 1. Correlations separately by dataset x keyset
# ------------------------------------------------------------

rows = []

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

        for feature in FEATURES:

            valid = (
                df[feature].notna()
                & delta.notna()
            )

            x = df.loc[
                valid,
                feature,
            ]

            y = delta.loc[
                valid
            ]

            if len(x) < 3:
                rho = np.nan
                p_value = np.nan

            elif (
                x.nunique() < 2
                or y.nunique() < 2
            ):
                rho = np.nan
                p_value = np.nan

            else:
                rho, p_value = spearmanr(
                    x,
                    y,
                )

            rows.append({
                "dataset": dataset,
                "keyset": keyset,
                "method_a": method_a,
                "method_b": method_b,
                "feature": feature,
                "n": len(x),
                "rho": rho,
                "p_value": p_value,
            })


results = pd.DataFrame(rows)

results.to_csv(
    DATA_DIR
    / "feature_advantage_correlations.csv",
    index=False,
)


# ------------------------------------------------------------
# 2. Put the three keysets side-by-side
# ------------------------------------------------------------

rho_wide = (
    results
    .pivot(
        index=[
            "dataset",
            "method_a",
            "method_b",
            "feature",
        ],
        columns="keyset",
        values="rho",
    )
    .reset_index()
)


rho_wide = rho_wide.rename(
    columns={
        0: "rho_keyset0",
        1: "rho_keyset1",
        2: "rho_keyset2",
    }
)


rho_columns = [
    "rho_keyset0",
    "rho_keyset1",
    "rho_keyset2",
]


rho_wide["mean_rho"] = (
    rho_wide[rho_columns]
    .mean(axis=1)
)

rho_wide["mean_abs_rho"] = (
    rho_wide[rho_columns]
    .abs()
    .mean(axis=1)
)

rho_wide["min_rho"] = (
    rho_wide[rho_columns]
    .min(axis=1)
)

rho_wide["max_rho"] = (
    rho_wide[rho_columns]
    .max(axis=1)
)

rho_wide["rho_range"] = (
    rho_wide["max_rho"]
    - rho_wide["min_rho"]
)


# ------------------------------------------------------------
# 3. Sign consistency
# ------------------------------------------------------------

def sign_label(row):

    values = row[rho_columns]

    if values.isna().any():
        return "missing"

    if (values > 0).all():
        return "+++"

    if (values < 0).all():
        return "---"

    return "mixed"


rho_wide["sign_pattern"] = (
    rho_wide.apply(
        sign_label,
        axis=1,
    )
)

rho_wide["consistent_signs"] = (
    rho_wide["sign_pattern"]
    .isin(["+++", "---"])
)


rho_wide["abs_mean_rho"] = (
    rho_wide["mean_rho"]
    .abs()
)


rho_wide.to_csv(
    DATA_DIR
    / "feature_advantage_correlation_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print(
    "\nStrongest same-sign "
    "feature-advantage associations:"
)

for dataset in master["dataset"].unique():

    print(f"\n--- {dataset} ---")

    top = (
        rho_wide[
            (rho_wide["dataset"] == dataset)
            & rho_wide["consistent_signs"]
        ]
        .sort_values(
            "abs_mean_rho",
            ascending=False,
        )
        .head(15)
    )

    print(
        top[
            [
                "method_a",
                "method_b",
                "feature",
                "rho_keyset0",
                "rho_keyset1",
                "rho_keyset2",
                "mean_rho",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


print("\nSaved:")
print(
    DATA_DIR
    / "feature_advantage_correlations.csv"
)
print(
    DATA_DIR
    / "feature_advantage_correlation_summary.csv"
)