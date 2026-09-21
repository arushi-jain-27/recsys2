from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

DATASETS = ["dunnhumby", "instacart", "sam", "tafeng", "valuedshopper"]
METHODS = ["recency", "frequency", "tifuknn", "fpmc", "triple2vec", "dnntsp", "cbp", "diffrec"]
FEATURES = ["transition_strength", "relational_strength"]
MIN_BASKETS = [2, 3, 5]


def spearman(x, y):
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return x.corr(y, method="spearman")


print(f"{datetime.now()}: loading tables", flush=True)

performance = pd.read_csv(DATA_DIR / "performance.csv", dtype={"user": str})
features = pd.read_csv(DATA_DIR / "regime_features.csv", dtype={"user": str})

performance = performance[performance["dataset"].isin(DATASETS)].copy()
features = features[features["dataset"].isin(DATASETS)].copy()

regimes = (
    performance[["dataset", "user"]]
    .drop_duplicates()
    .merge(
        features[["dataset", "user", "n_baskets"] + FEATURES],
        on=["dataset", "user"],
        how="left",
        validate="one_to_one",
    )
)

analysis = performance.merge(regimes, on=["dataset", "user"], how="left", validate="many_to_one")
analysis["user_mean_ndcg"] = analysis[METHODS].mean(axis=1)

for method in METHODS:
    analysis[f"{method}_centered"] = analysis[method] - analysis["user_mean_ndcg"]


print(f"{datetime.now()}: computing reliability slices", flush=True)

rows = []

for min_baskets in MIN_BASKETS:
    deep = analysis[analysis["n_baskets"] >= min_baskets]

    for (dataset, keyset), df in deep.groupby(["dataset", "keyset"]):
        n_slice = df["user"].nunique()

        for feature in FEATURES:
            for method in METHODS:
                valid = df[feature].notna() & df[method].notna()
                x = df.loc[valid, feature]
                y_raw = df.loc[valid, method]
                y_centered = df.loc[valid, f"{method}_centered"]

                rows.append({
                    "dataset": dataset,
                    "keyset": keyset,
                    "feature": feature,
                    "min_baskets": min_baskets,
                    "method": method,
                    "n_users_slice": n_slice,
                    "n_users_feature": len(x),
                    "rho_raw": spearman(x, y_raw),
                    "rho_centered": spearman(x, y_centered),
                })


effects = pd.DataFrame(rows)

summary = (
    effects
    .groupby(["dataset", "feature", "min_baskets", "method"], as_index=False)
    .agg(
        n_users_slice=("n_users_slice", "mean"),
        n_users_feature=("n_users_feature", "mean"),
        rho_raw_mean=("rho_raw", "mean"),
        rho_raw_sd=("rho_raw", "std"),
        rho_centered_mean=("rho_centered", "mean"),
        rho_centered_sd=("rho_centered", "std"),
        n_keysets=("rho_centered", "count"),
    )
)


def sign_label(series):
    values = series.dropna()
    if len(values) < 3:
        return "missing"
    if (values > 0).all():
        return "positive"
    if (values < 0).all():
        return "negative"
    return "mixed"


signs = (
    effects
    .groupby(["dataset", "feature", "min_baskets", "method"])["rho_centered"]
    .apply(sign_label)
    .reset_index(name="rho_centered_sign")
)

summary = summary.merge(
    signs,
    on=["dataset", "feature", "min_baskets", "method"],
    how="left",
)

coverage = (
    regimes
    .groupby("dataset")
    .apply(
        lambda df: pd.Series({
            "n_eval": len(df),
            "n_ge2": int((df["n_baskets"] >= 2).sum()),
            "n_ge3": int((df["n_baskets"] >= 3).sum()),
            "n_ge5": int((df["n_baskets"] >= 5).sum()),
            "transition_missing": df["transition_strength"].isna().mean(),
            "relational_missing": df["relational_strength"].isna().mean(),
        })
    )
    .reset_index()
)

output_path = DATA_DIR / "association_reliability.csv"
summary.to_csv(output_path, index=False)
coverage.to_csv(DATA_DIR / "association_reliability_coverage.csv", index=False)


print("\nEvaluated-user coverage by history depth:")
print(
    coverage
    .assign(
        frac_ge2=lambda d: d["n_ge2"] / d["n_eval"],
        frac_ge3=lambda d: d["n_ge3"] / d["n_eval"],
        frac_ge5=lambda d: d["n_ge5"] / d["n_eval"],
    )
    [
        [
            "dataset",
            "n_eval",
            "frac_ge2",
            "frac_ge3",
            "frac_ge5",
            "transition_missing",
            "relational_missing",
        ]
    ]
    .round(3)
    .to_string(index=False)
)

print("\nCentered Spearman by history-depth slice")
print("(recomputed on remaining 8 methods; RACF/Sets2Sets excluded from the user mean)\n")

show_methods = ["frequency", "tifuknn", "fpmc", "dnntsp", "cbp", "triple2vec"]

for dataset in DATASETS:
    print(f"\n=== {dataset} ===")

    for feature in FEATURES:
        block = (
            summary[
                (summary["dataset"] == dataset)
                & (summary["feature"] == feature)
                & (summary["method"].isin(show_methods))
            ]
            .pivot(index="method", columns="min_baskets", values="rho_centered_mean")
            .reindex(show_methods)
            .reindex(columns=MIN_BASKETS)
        )
        signs_block = (
            summary[
                (summary["dataset"] == dataset)
                & (summary["feature"] == feature)
                & (summary["method"].isin(show_methods))
            ]
            .pivot(index="method", columns="min_baskets", values="rho_centered_sign")
            .reindex(show_methods)
            .reindex(columns=MIN_BASKETS)
        )

        print(f"\n{feature}: rho_centered_mean at n_baskets >= 2 / 3 / 5")
        print(block.round(3).to_string())
        print("sign pattern:")
        print(signs_block.to_string())


print("\nDoes |rho_centered| increase as the history-depth floor rises?")
print("Compared  >=2  vs  >=5  on the same method/dataset/feature.\n")

compare_rows = []

for (dataset, feature, method), df in summary.groupby(["dataset", "feature", "method"]):
    if method not in show_methods:
        continue

    by_t = df.set_index("min_baskets")["rho_centered_mean"]
    if 2 not in by_t.index or 5 not in by_t.index:
        continue

    rho2, rho5 = by_t.loc[2], by_t.loc[5]
    compare_rows.append({
        "dataset": dataset,
        "feature": feature,
        "method": method,
        "rho_ge2": rho2,
        "rho_ge5": rho5,
        "abs_ge2": abs(rho2) if pd.notna(rho2) else np.nan,
        "abs_ge5": abs(rho5) if pd.notna(rho5) else np.nan,
        "abs_change": (abs(rho5) - abs(rho2)) if pd.notna(rho2) and pd.notna(rho5) else np.nan,
    })

compare = pd.DataFrame(compare_rows)

for feature in FEATURES:
    print(f"\n--- {feature}: |rho| change from >=2 to >=5 ---")
    top = (
        compare[compare["feature"] == feature]
        .sort_values("abs_change", ascending=False)
    )
    print(
        top[
            ["dataset", "method", "rho_ge2", "rho_ge5", "abs_change"]
        ]
        .round(3)
        .to_string(index=False)
    )

print("\nRead this as:")
print("  |rho| rises with depth  -> short histories were attenuating a real signal")
print("  |rho| stays flat        -> measurement error was not the main issue")
print("  |rho| shrinks / flips   -> the shallow-history association was not reliable")
print("  Do not treat n_baskets>=5 as the paper population.")
print("\nSaved:")
print(output_path)
print(DATA_DIR / "association_reliability_coverage.csv")
