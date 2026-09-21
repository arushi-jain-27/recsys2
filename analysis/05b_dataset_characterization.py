from pathlib import Path

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


# ------------------------------------------------------------
# Load master table
# ------------------------------------------------------------

master = pd.read_csv(
    DATA_DIR / "master_analysis.csv",
    dtype={"user": str},
)


# Features are fixed for a user across keysets.
# Keep one row per dataset x user.
users = (
    master[
        ["dataset", "user"] + FEATURES
    ]
    .drop_duplicates(["dataset", "user"])
    .copy()
)


print("\nUnique users by dataset:")
print(users.groupby("dataset").size())


# ------------------------------------------------------------
# 1. Dataset-level feature summaries
# ------------------------------------------------------------

rows = []

for dataset, df in users.groupby("dataset"):

    for feature in FEATURES:

        values = df[feature].dropna()

        rows.append({
            "dataset": dataset,
            "feature": feature,
            "n": len(values),
            "missing_rate": df[feature].isna().mean(),
            "p10": values.quantile(0.10),
            "median": values.median(),
            "p90": values.quantile(0.90),
        })


summary = pd.DataFrame(rows)

summary.to_csv(
    DATA_DIR / "dataset_feature_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# 2. Dataset x feature median table
# ------------------------------------------------------------

median_profile = (
    summary
    .pivot(
        index="dataset",
        columns="feature",
        values="median",
    )
    [FEATURES]
)

median_profile.to_csv(
    DATA_DIR / "dataset_feature_medians.csv"
)


# ------------------------------------------------------------
# 3. Standardize dataset medians feature-by-feature
#
# This is only for comparing behavioral profiles across
# datasets / eventual heatmap visualization.
# ------------------------------------------------------------

profile_z = median_profile.copy()

for feature in FEATURES:

    values = median_profile[feature]

    mean = values.mean()
    std = values.std(ddof=0)

    if std == 0 or np.isnan(std):
        profile_z[feature] = 0.0
    else:
        profile_z[feature] = (
            values - mean
        ) / std


profile_z.to_csv(
    DATA_DIR / "dataset_feature_profile_z.csv"
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print("\nDataset feature medians:")
print(
    median_profile
    .round(3)
    .to_string()
)


print("\nStandardized dataset behavioral profiles:")
print(
    profile_z
    .round(2)
    .to_string()
)


print("\nSaved:")
print(DATA_DIR / "dataset_feature_summary.csv")
print(DATA_DIR / "dataset_feature_medians.csv")
print(DATA_DIR / "dataset_feature_profile_z.csv")