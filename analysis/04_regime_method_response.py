from pathlib import Path
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
PRIMARY_DATASETS = ["dunnhumby", "instacart", "sam", "tafeng", "valuedshopper"]
N_BINS = 5

performance = pd.read_csv(DATA_DIR / "performance.csv", dtype={"user": str})
features = pd.read_csv(DATA_DIR / "regime_features.csv", dtype={"user": str})

assert not performance.duplicated(["dataset", "keyset", "user"]).any()
assert not features.duplicated(["dataset", "user"]).any()

analysis_users = performance[["dataset", "user"]].drop_duplicates()
regimes = analysis_users.merge(features[["dataset", "user"] + FEATURES], on=["dataset", "user"], how="left", validate="one_to_one")
assert len(regimes) == len(analysis_users)

# Fixed dataset-level bins: a user gets the same bin in every keyset.
# Percentile ranks preserve ties, so discrete features may occupy fewer than 5 bins.
for feature in FEATURES:
    regimes[f"{feature}_bin"] = np.nan

    for dataset, idx in regimes.groupby("dataset").groups.items():
        x = regimes.loc[idx, feature]
        valid = x.notna()

        if valid.sum() == 0:
            continue

        pct_rank = x.loc[valid].rank(method="average", pct=True)
        bins = np.ceil(pct_rank * N_BINS).clip(1, N_BINS).astype(int)
        regimes.loc[pct_rank.index, f"{feature}_bin"] = bins

analysis = performance.merge(regimes, on=["dataset", "user"], how="left", validate="many_to_one")
analysis["user_mean_ndcg"] = analysis[METHODS].mean(axis=1)

for method in METHODS:
    analysis[f"{method}_centered"] = analysis[method] - analysis["user_mean_ndcg"]


# ------------------------------------------------------------------
# 1. Regime response curves
# ------------------------------------------------------------------

response_rows = []

for (dataset, keyset), df in analysis.groupby(["dataset", "keyset"]):
    for feature in FEATURES:
        bin_col = f"{feature}_bin"
        valid = df[feature].notna() & df[bin_col].notna()
        temp = df.loc[valid].copy()

        for bin_id, bin_df in temp.groupby(bin_col):
            for method in METHODS:
                response_rows.append({
                    "dataset": dataset,
                    "keyset": keyset,
                    "feature": feature,
                    "bin": int(bin_id),
                    "method": method,
                    "n_users": len(bin_df),
                    "feature_min": bin_df[feature].min(),
                    "feature_median": bin_df[feature].median(),
                    "feature_max": bin_df[feature].max(),
                    "mean_ndcg": bin_df[method].mean(),
                    "mean_centered_ndcg": bin_df[f"{method}_centered"].mean(),
                })

response = pd.DataFrame(response_rows)
response.to_csv(DATA_DIR / "regime_method_response.csv", index=False)


# ------------------------------------------------------------------
# 2. Continuous trend + low-to-high effect
# ------------------------------------------------------------------

effect_rows = []

for (dataset, keyset), df in analysis.groupby(["dataset", "keyset"]):
    for feature in FEATURES:
        bin_col = f"{feature}_bin"

        for method in METHODS:
            valid = df[feature].notna() & df[method].notna()
            x = df.loc[valid, feature]
            y_raw = df.loc[valid, method]
            y_centered = df.loc[valid, f"{method}_centered"]

            rho_raw = x.corr(y_raw, method="spearman") if len(x) >= 3 and x.nunique() >= 2 and y_raw.nunique() >= 2 else np.nan
            rho_centered = x.corr(y_centered, method="spearman") if len(x) >= 3 and x.nunique() >= 2 and y_centered.nunique() >= 2 else np.nan

            method_response = response[
                (response["dataset"] == dataset)
                & (response["keyset"] == keyset)
                & (response["feature"] == feature)
                & (response["method"] == method)
            ].sort_values("bin")

            if len(method_response) >= 2:
                low, high = method_response.iloc[0], method_response.iloc[-1]
                raw_low_to_high = high["mean_ndcg"] - low["mean_ndcg"]
                centered_low_to_high = high["mean_centered_ndcg"] - low["mean_centered_ndcg"]
                n_bins = len(method_response)
            else:
                raw_low_to_high = np.nan
                centered_low_to_high = np.nan
                n_bins = len(method_response)

            effect_rows.append({
                "dataset": dataset,
                "keyset": keyset,
                "feature": feature,
                "method": method,
                "n_users": len(x),
                "n_bins": n_bins,
                "rho_raw": rho_raw,
                "rho_centered": rho_centered,
                "raw_low_to_high": raw_low_to_high,
                "centered_low_to_high": centered_low_to_high,
            })

effects_by_keyset = pd.DataFrame(effect_rows)


# ------------------------------------------------------------------
# 3. Put the three keysets side by side
# ------------------------------------------------------------------

index_cols = ["dataset", "feature", "method"]
summary = effects_by_keyset[index_cols].drop_duplicates().copy()

for metric in ["rho_raw", "rho_centered", "raw_low_to_high", "centered_low_to_high"]:
    wide = effects_by_keyset.pivot(index=index_cols, columns="keyset", values=metric).reset_index()

    for keyset in sorted(performance["keyset"].unique()):
        if keyset not in wide.columns:
            wide[keyset] = np.nan

    wide = wide.rename(columns={keyset: f"{metric}_k{keyset}" for keyset in sorted(performance["keyset"].unique())})
    summary = summary.merge(wide, on=index_cols, how="left", validate="one_to_one")

for metric in ["rho_raw", "rho_centered", "raw_low_to_high", "centered_low_to_high"]:
    cols = [c for c in summary.columns if c.startswith(f"{metric}_k")]
    summary[f"{metric}_mean"] = summary[cols].mean(axis=1)
    summary[f"{metric}_sd"] = summary[cols].std(axis=1, ddof=0)


def sign_consistency(row, metric):
    cols = [c for c in row.index if c.startswith(f"{metric}_k")]
    values = row[cols].dropna()

    if len(values) < 3:
        return "missing"
    if (values > 0).all():
        return "positive"
    if (values < 0).all():
        return "negative"
    return "mixed"


summary["rho_raw_sign"] = summary.apply(lambda row: sign_consistency(row, "rho_raw"), axis=1)
summary["rho_centered_sign"] = summary.apply(lambda row: sign_consistency(row, "rho_centered"), axis=1)
summary["centered_effect_sign"] = summary.apply(lambda row: sign_consistency(row, "centered_low_to_high"), axis=1)

summary["abs_rho_centered_mean"] = summary["rho_centered_mean"].abs()
summary["abs_centered_low_to_high_mean"] = summary["centered_low_to_high_mean"].abs()
summary = summary.sort_values(["dataset", "abs_rho_centered_mean"], ascending=[True, False])
summary.to_csv(DATA_DIR / "regime_method_effects.csv", index=False)


# ------------------------------------------------------------------
# Console diagnostics
# ------------------------------------------------------------------

print("\nSaved:")
print(DATA_DIR / "regime_method_response.csv")
print(DATA_DIR / "regime_method_effects.csv")

print("\nStrongest split-stable relative method responses on primary datasets:")

stable = summary[
    summary["dataset"].isin(PRIMARY_DATASETS)
    & (summary["rho_centered_sign"].isin(["positive", "negative"]))
].copy()

for dataset in PRIMARY_DATASETS:
    print(f"\n--- {dataset} ---")
    top = stable[stable["dataset"] == dataset].sort_values("abs_rho_centered_mean", ascending=False).head(12)
    print(
        top[
            [
                "feature",
                "method",
                "rho_centered_mean",
                "rho_centered_sd",
                "centered_low_to_high_mean",
                "centered_low_to_high_sd",
                "rho_centered_sign",
            ]
        ].round(4).to_string(index=False)
    )

print("\nInterpretation:")
print("  mean_ndcg: absolute method performance across the regime.")
print("  mean_centered_ndcg: method performance relative to the same user's 10-method average.")
print("  rho_centered_mean: whether a method becomes relatively stronger/weaker as the regime feature increases.")
print("  centered_low_to_high_mean: relative NDCG change from the lowest occupied bin to the highest occupied bin.")
