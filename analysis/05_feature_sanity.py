from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

FEATURES = [
    "n_baskets",
    "avg_basket_size",
    "repeat_rate",
    "adjacent_basket_jaccard",
    "recent_basket_novelty",
    "popularity_profile",
    "neighbor_similarity",
    "sequential_specificity",
    "relational_strength",
]


master = pd.read_csv(
    DATA_DIR / "master_analysis.csv",
    dtype={"user": str},
)

# One feature vector per actual analysis user
users = (
    master[["dataset", "user"] + FEATURES]
    .drop_duplicates(["dataset", "user"])
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 1. Feature summaries
# ------------------------------------------------------------

summary_rows = []

for dataset, df in users.groupby("dataset"):
    for feature in FEATURES:
        x = df[feature].dropna()

        summary_rows.append({
            "dataset": dataset,
            "feature": feature,
            "count": len(x),
            "mean": x.mean(),
            "std": x.std(),
            "min": x.min(),
            "p10": x.quantile(0.10),
            "median": x.median(),
            "p90": x.quantile(0.90),
            "max": x.max(),
        })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    DATA_DIR / "feature_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# 2. Missingness
# ------------------------------------------------------------

missing_rows = []

for dataset, df in users.groupby("dataset"):
    for feature in FEATURES:
        missing_rows.append({
            "dataset": dataset,
            "feature": feature,
            "n_users": len(df),
            "n_missing": df[feature].isna().sum(),
            "missing_rate": df[feature].isna().mean(),
        })

missingness = pd.DataFrame(missing_rows)

missingness.to_csv(
    DATA_DIR / "feature_missingness.csv",
    index=False,
)


# ------------------------------------------------------------
# 3. Within-dataset Spearman correlations
# ------------------------------------------------------------

correlation_rows = []

for dataset, df in users.groupby("dataset"):
    corr = df[FEATURES].corr(method="spearman")

    for feature_a, feature_b in combinations(FEATURES, 2):
        value = corr.loc[feature_a, feature_b]

        correlation_rows.append({
            "dataset": dataset,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "spearman": value,
            "abs_spearman": abs(value),
        })

correlations = pd.DataFrame(correlation_rows)

correlations.to_csv(
    DATA_DIR / "feature_correlations.csv",
    index=False,
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print("\nUnique analysis users:")
print(users.groupby("dataset").size())

print("\nMissingness rates by dataset:")
print(
    missingness
    .pivot(index="feature", columns="dataset", values="missing_rate")
    .round(3)
)

print("\nStrongest feature correlations by dataset:")

for dataset in users["dataset"].unique():
    print(f"\n--- {dataset} ---")

    top = (
        correlations[correlations["dataset"] == dataset]
        .dropna()
        .sort_values("abs_spearman", ascending=False)
        .head(10)
    )

    print(
        top[
            ["feature_a", "feature_b", "spearman"]
        ].to_string(index=False)
    )

print("\nFeature ranges / medians:")
print(
    summary[
        ["dataset", "feature", "p10", "median", "p90"]
    ].to_string(index=False)
)

print("\nSaved:")
print(DATA_DIR / "feature_summary.csv")
print(DATA_DIR / "feature_missingness.csv")
print(DATA_DIR / "feature_correlations.csv")