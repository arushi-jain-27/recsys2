from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

METHODS = ["recency", "frequency", "racf", "tifuknn", "fpmc", "triple2vec", "sets2sets", "dnntsp", "cbp", "diffrec"]
FEATURES = [
    "n_baskets",
    "avg_basket_size",
    "repeat_rate",
    "adjacent_basket_jaccard",
    "recent_basket_novelty",
    "popularity_profile",
    "neighbor_similarity",
    "transition_strength",
    "relational_strength",
]
EPS = 1e-12

performance = pd.read_csv(DATA_DIR / "performance.csv", dtype={"user": str})
features = pd.read_csv(DATA_DIR / "regime_features.csv", dtype={"user": str})

assert not performance.duplicated(["dataset", "keyset", "user"]).any()
assert not features.duplicated(["dataset", "user"]).any()

analysis_users = performance[["dataset", "user"]].drop_duplicates()
regimes = analysis_users.merge(features, on=["dataset", "user"], how="left", validate="one_to_one")

assert len(regimes) == len(analysis_users)
assert not regimes[FEATURES].isna().all(axis=1).any(), "Some evaluated users are missing all regime features"


# Regime characterization
summary_rows = []

for dataset, df in regimes.groupby("dataset"):
    for feature in FEATURES:
        x = df[feature].dropna()

        summary_rows.append({
            "dataset": dataset,
            "feature": feature,
            "n_users": len(df),
            "n_nonmissing": len(x),
            "missing_rate": df[feature].isna().mean(),
            "n_unique": x.nunique(),
            "zero_rate": (x == 0).mean() if len(x) else np.nan,
            "mean": x.mean(),
            "std": x.std(),
            "min": x.min(),
            "p10": x.quantile(0.10),
            "median": x.median(),
            "p90": x.quantile(0.90),
            "max": x.max(),
        })

regime_summary = pd.DataFrame(summary_rows)
regime_summary.to_csv(DATA_DIR / "regime_summary.csv", index=False)

correlation_rows = []

for dataset, df in regimes.groupby("dataset"):
    corr = df[FEATURES].corr(method="spearman")

    for feature_a, feature_b in combinations(FEATURES, 2):
        rho = corr.loc[feature_a, feature_b]

        correlation_rows.append({
            "dataset": dataset,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "spearman": rho,
            "abs_spearman": abs(rho) if pd.notna(rho) else np.nan,
        })

regime_correlations = pd.DataFrame(correlation_rows)
regime_correlations.to_csv(DATA_DIR / "regime_correlations.csv", index=False)


# Benchmark characterization
benchmark_by_keyset = performance.groupby(["dataset", "keyset"])[METHODS].mean().reset_index()
benchmark_by_keyset.to_csv(DATA_DIR / "benchmark_by_keyset.csv", index=False)

benchmark_rows = []

for dataset, df in benchmark_by_keyset.groupby("dataset"):
    for method in METHODS:
        x = df[method].dropna()

        benchmark_rows.append({
            "dataset": dataset,
            "method": method,
            "n_keysets": len(x),
            "mean_ndcg": x.mean(),
            "keyset_sd": x.std(ddof=0),
            "min_ndcg": x.min(),
            "max_ndcg": x.max(),
        })

benchmark_summary = pd.DataFrame(benchmark_rows)
benchmark_summary["rank"] = benchmark_summary.groupby("dataset")["mean_ndcg"].rank(method="min", ascending=False).astype(int)
benchmark_summary = benchmark_summary.sort_values(["dataset", "rank", "method"])
benchmark_summary.to_csv(DATA_DIR / "benchmark_summary.csv", index=False)

diagnostic_rows = []

for (dataset, keyset), df in performance.groupby(["dataset", "keyset"]):
    all_zero = df[METHODS].abs().max(axis=1) <= EPS

    diagnostic_rows.append({
        "dataset": dataset,
        "keyset": keyset,
        "n_users": len(df),
        "n_all_zero": int(all_zero.sum()),
        "all_zero_rate": all_zero.mean(),
    })

evaluation_diagnostics = pd.DataFrame(diagnostic_rows)
evaluation_diagnostics.to_csv(DATA_DIR / "evaluation_diagnostics.csv", index=False)


# Console output
print("\nRegime feature medians:")
print(regime_summary.pivot(index="dataset", columns="feature", values="median")[FEATURES].round(3).to_string())

print("\nMissingness:")
print(regime_summary.pivot(index="dataset", columns="feature", values="missing_rate")[FEATURES].round(3).to_string())

print("\nStrongest feature correlations:")
for dataset in regimes["dataset"].unique():
    top = regime_correlations[regime_correlations["dataset"] == dataset].dropna().sort_values("abs_spearman", ascending=False).head(10)
    print(f"\n--- {dataset} ---")
    print(top[["feature_a", "feature_b", "spearman"]].round(3).to_string(index=False))

print("\nBenchmark performance averaged across keysets:")
print(benchmark_summary.pivot(index="dataset", columns="method", values="mean_ndcg")[METHODS].round(4).to_string())

print("\nBest method by dataset:")
print(benchmark_summary[benchmark_summary["rank"] == 1][["dataset", "method", "mean_ndcg", "keyset_sd"]].round(4).to_string(index=False))

print("\nAll-zero evaluation rate averaged across keysets:")
print(evaluation_diagnostics.groupby("dataset")["all_zero_rate"].mean().round(3).to_string())

print("\nSaved:")
for filename in ["regime_summary.csv", "regime_correlations.csv", "benchmark_by_keyset.csv", "benchmark_summary.csv", "evaluation_diagnostics.csv"]:
    print(DATA_DIR / filename)
