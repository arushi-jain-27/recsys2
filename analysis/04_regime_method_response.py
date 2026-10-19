from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

METHODS = ["recency", "frequency", "tifuknn", "fpmc", "triple2vec", "dnntsp", "cbp", "diffrec"]
FEATURES = ["n_baskets", "avg_basket_size", "repeat_rate", "adjacent_basket_jaccard", "recent_basket_novelty", "popularity_profile", "neighbor_similarity", "transition_strength", "relational_strength"]
PRIMARY_DATASETS = ["dunnhumby", "instacart", "sam", "tafeng", "valuedshopper"]
N_BINS = 5
EPS = 1e-12

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
        if valid.sum() == 0: continue
        pct_rank = x.loc[valid].rank(method="average", pct=True)
        bins = np.ceil(pct_rank * N_BINS).clip(1, N_BINS).astype(int)
        regimes.loc[pct_rank.index, f"{feature}_bin"] = bins

analysis = performance.merge(regimes, on=["dataset", "user"], how="left", validate="many_to_one")
analysis["user_mean_ndcg"] = analysis[METHODS].mean(axis=1)
for method in METHODS: analysis[f"{method}_centered"] = analysis[method] - analysis["user_mean_ndcg"]

# Global single-best method per dataset: first compute mean performance within each keyset,
# then average keyset means so each keyset receives equal descriptive weight.
global_method_means = performance.groupby(["dataset", "keyset"])[METHODS].mean().groupby("dataset").mean()
global_best_by_dataset = global_method_means.idxmax(axis=1).to_dict()

print("\nGlobal single-best method by dataset:")
for dataset, method in global_best_by_dataset.items(): print(f"  {dataset}: {method}")


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
                    "dataset": dataset, "keyset": keyset, "feature": feature, "bin": int(bin_id), "method": method, "n_users": len(bin_df),
                    "feature_min": bin_df[feature].min(), "feature_median": bin_df[feature].median(), "feature_max": bin_df[feature].max(),
                    "mean_ndcg": bin_df[method].mean(), "mean_centered_ndcg": bin_df[f"{method}_centered"].mean(),
                })

response = pd.DataFrame(response_rows)
response.to_csv(DATA_DIR / "regime_method_response.csv", index=False)


# ------------------------------------------------------------------
# 2. Continuous trend + low-to-high effect
# ------------------------------------------------------------------

effect_rows = []

for (dataset, keyset), df in analysis.groupby(["dataset", "keyset"]):
    for feature in FEATURES:
        for method in METHODS:
            valid = df[feature].notna() & df[method].notna()
            x = df.loc[valid, feature]
            y_raw = df.loc[valid, method]
            y_centered = df.loc[valid, f"{method}_centered"]

            rho_raw = x.corr(y_raw, method="spearman") if len(x) >= 3 and x.nunique() >= 2 and y_raw.nunique() >= 2 else np.nan
            rho_centered = x.corr(y_centered, method="spearman") if len(x) >= 3 and x.nunique() >= 2 and y_centered.nunique() >= 2 else np.nan

            method_response = response[
                (response["dataset"] == dataset) & (response["keyset"] == keyset) &
                (response["feature"] == feature) & (response["method"] == method)
            ].sort_values("bin")

            if len(method_response) >= 2:
                low, high = method_response.iloc[0], method_response.iloc[-1]
                raw_low_to_high = high["mean_ndcg"] - low["mean_ndcg"]
                centered_low_to_high = high["mean_centered_ndcg"] - low["mean_centered_ndcg"]
                n_bins = len(method_response)
            else:
                raw_low_to_high = centered_low_to_high = np.nan
                n_bins = len(method_response)

            effect_rows.append({
                "dataset": dataset, "keyset": keyset, "feature": feature, "method": method, "n_users": len(x), "n_bins": n_bins,
                "rho_raw": rho_raw, "rho_centered": rho_centered, "raw_low_to_high": raw_low_to_high, "centered_low_to_high": centered_low_to_high,
            })

effects_by_keyset = pd.DataFrame(effect_rows)


# ------------------------------------------------------------------
# 3. Put the three keysets side by side
# ------------------------------------------------------------------

index_cols = ["dataset", "feature", "method"]
summary = effects_by_keyset[index_cols].drop_duplicates().copy()
keysets = sorted(performance["keyset"].unique())

for metric in ["rho_raw", "rho_centered", "raw_low_to_high", "centered_low_to_high"]:
    wide = effects_by_keyset.pivot(index=index_cols, columns="keyset", values=metric).reset_index()
    for keyset in keysets:
        if keyset not in wide.columns: wide[keyset] = np.nan
    wide = wide.rename(columns={keyset: f"{metric}_k{keyset}" for keyset in keysets})
    summary = summary.merge(wide, on=index_cols, how="left", validate="one_to_one")

for metric in ["rho_raw", "rho_centered", "raw_low_to_high", "centered_low_to_high"]:
    cols = [c for c in summary.columns if c.startswith(f"{metric}_k")]
    summary[f"{metric}_mean"] = summary[cols].mean(axis=1)
    summary[f"{metric}_sd"] = summary[cols].std(axis=1, ddof=0)

def sign_consistency(row, metric):
    cols = [c for c in row.index if c.startswith(f"{metric}_k")]
    values = row[cols].dropna()
    if len(values) < 3: return "missing"
    if (values > 0).all(): return "positive"
    if (values < 0).all(): return "negative"
    return "mixed"

summary["rho_raw_sign"] = summary.apply(lambda row: sign_consistency(row, "rho_raw"), axis=1)
summary["rho_centered_sign"] = summary.apply(lambda row: sign_consistency(row, "rho_centered"), axis=1)
summary["centered_effect_sign"] = summary.apply(lambda row: sign_consistency(row, "centered_low_to_high"), axis=1)
summary["abs_rho_centered_mean"] = summary["rho_centered_mean"].abs()
summary["abs_centered_low_to_high_mean"] = summary["centered_low_to_high_mean"].abs()
summary = summary.sort_values(["dataset", "abs_rho_centered_mean"], ascending=[True, False])
summary.to_csv(DATA_DIR / "regime_method_effects.csv", index=False)


# ------------------------------------------------------------------
# 4. Challenger-vs-global-best regime response
# ------------------------------------------------------------------

contrast_response_rows = []

for (dataset, keyset), df in analysis.groupby(["dataset", "keyset"]):
    global_method = global_best_by_dataset[dataset]

    for feature in FEATURES:
        bin_col = f"{feature}_bin"
        valid = df[feature].notna() & df[bin_col].notna()
        temp = df.loc[valid].copy()

        for bin_id, bin_df in temp.groupby(bin_col):
            global_scores = bin_df[global_method].to_numpy(float)
            any_method_positive = bin_df[METHODS].max(axis=1).to_numpy(float) > EPS
            n_nonzero_users = int(any_method_positive.sum())

            for method in METHODS:
                if method == global_method: continue

                challenger_scores = bin_df[method].to_numpy(float)
                delta = challenger_scores - global_scores
                challenger_better = delta > EPS
                global_better = delta < -EPS
                tied = ~(challenger_better | global_better)
                pair_active = (challenger_scores > EPS) | (global_scores > EPS)
                n_pair_active = int(pair_active.sum())

                contrast_response_rows.append({
                    "dataset": dataset, "keyset": keyset, "feature": feature, "bin": int(bin_id), "method": method, "global_method": global_method,
                    "n_users": len(bin_df), "n_nonzero_users": n_nonzero_users, "n_pair_active": n_pair_active,
                    "feature_min": bin_df[feature].min(), "feature_median": bin_df[feature].median(), "feature_max": bin_df[feature].max(),
                    "mean_method_ndcg": challenger_scores.mean(), "mean_global_ndcg": global_scores.mean(), "mean_delta_ndcg": delta.mean(),
                    "mean_delta_ndcg_nonzero_users": delta[any_method_positive].mean() if n_nonzero_users else np.nan,
                    "mean_delta_ndcg_pair_active": delta[pair_active].mean() if n_pair_active else np.nan,
                    "challenger_win_rate_all": challenger_better.mean(), "global_win_rate_all": global_better.mean(), "tie_rate_all": tied.mean(),
                    "challenger_win_rate_nonzero_users": challenger_better[any_method_positive].mean() if n_nonzero_users else np.nan,
                    "global_win_rate_nonzero_users": global_better[any_method_positive].mean() if n_nonzero_users else np.nan,
                    "tie_rate_nonzero_users": tied[any_method_positive].mean() if n_nonzero_users else np.nan,
                    "challenger_win_rate_pair_active": challenger_better[pair_active].mean() if n_pair_active else np.nan,
                    "global_win_rate_pair_active": global_better[pair_active].mean() if n_pair_active else np.nan,
                    "tie_rate_pair_active": tied[pair_active].mean() if n_pair_active else np.nan,
                })

contrast_response = pd.DataFrame(contrast_response_rows)
contrast_response.to_csv(DATA_DIR / "regime_global_contrast_response.csv", index=False)


# ------------------------------------------------------------------
# 5. Continuous challenger-vs-global-best effects
# ------------------------------------------------------------------

contrast_effect_rows = []

for (dataset, keyset), df in analysis.groupby(["dataset", "keyset"]):
    global_method = global_best_by_dataset[dataset]

    for feature in FEATURES:
        for method in METHODS:
            if method == global_method: continue

            valid = df[feature].notna() & df[method].notna() & df[global_method].notna()
            temp = df.loc[valid, [feature] + METHODS].copy()
            x = temp[feature]
            delta = temp[method] - temp[global_method]
            nonzero_users = temp[METHODS].max(axis=1) > EPS

            rho_delta = x.corr(delta, method="spearman") if len(x) >= 3 and x.nunique() >= 2 and delta.nunique() >= 2 else np.nan
            x_nonzero, delta_nonzero = x[nonzero_users], delta[nonzero_users]
            rho_delta_nonzero_users = x_nonzero.corr(delta_nonzero, method="spearman") if len(x_nonzero) >= 3 and x_nonzero.nunique() >= 2 and delta_nonzero.nunique() >= 2 else np.nan

            method_response = contrast_response[
                (contrast_response["dataset"] == dataset) & (contrast_response["keyset"] == keyset) &
                (contrast_response["feature"] == feature) & (contrast_response["method"] == method)
            ].sort_values("bin")

            if len(method_response) >= 2:
                low, high = method_response.iloc[0], method_response.iloc[-1]
                delta_low_to_high = high["mean_delta_ndcg"] - low["mean_delta_ndcg"]
                delta_nonzero_low_to_high = high["mean_delta_ndcg_nonzero_users"] - low["mean_delta_ndcg_nonzero_users"]
                delta_pair_active_low_to_high = high["mean_delta_ndcg_pair_active"] - low["mean_delta_ndcg_pair_active"]
                challenger_win_rate_low = low["challenger_win_rate_nonzero_users"]
                challenger_win_rate_high = high["challenger_win_rate_nonzero_users"]
                challenger_win_rate_low_to_high = challenger_win_rate_high - challenger_win_rate_low
                challenger_win_rate_pair_active_low = low["challenger_win_rate_pair_active"]
                challenger_win_rate_pair_active_high = high["challenger_win_rate_pair_active"]
                challenger_win_rate_pair_active_low_to_high = challenger_win_rate_pair_active_high - challenger_win_rate_pair_active_low
                n_bins = len(method_response)
            else:
                delta_low_to_high = delta_nonzero_low_to_high = delta_pair_active_low_to_high = np.nan
                challenger_win_rate_low = challenger_win_rate_high = challenger_win_rate_low_to_high = np.nan
                challenger_win_rate_pair_active_low = challenger_win_rate_pair_active_high = challenger_win_rate_pair_active_low_to_high = np.nan
                n_bins = len(method_response)

            contrast_effect_rows.append({
                "dataset": dataset, "keyset": keyset, "feature": feature, "method": method, "global_method": global_method, "n_users": len(temp),
                "n_nonzero_users": int(nonzero_users.sum()), "n_bins": n_bins, "rho_delta": rho_delta, "rho_delta_nonzero_users": rho_delta_nonzero_users,
                "delta_low_to_high": delta_low_to_high, "delta_nonzero_low_to_high": delta_nonzero_low_to_high,
                "delta_pair_active_low_to_high": delta_pair_active_low_to_high,
                "challenger_win_rate_low": challenger_win_rate_low, "challenger_win_rate_high": challenger_win_rate_high,
                "challenger_win_rate_low_to_high": challenger_win_rate_low_to_high,
                "challenger_win_rate_pair_active_low": challenger_win_rate_pair_active_low,
                "challenger_win_rate_pair_active_high": challenger_win_rate_pair_active_high,
                "challenger_win_rate_pair_active_low_to_high": challenger_win_rate_pair_active_low_to_high,
            })

contrast_effects_by_keyset = pd.DataFrame(contrast_effect_rows)


# ------------------------------------------------------------------
# 6. Put global-best contrast effects across keysets side by side
# ------------------------------------------------------------------

contrast_index_cols = ["dataset", "feature", "method", "global_method"]
contrast_summary = contrast_effects_by_keyset[contrast_index_cols].drop_duplicates().copy()

contrast_metrics = [
    "rho_delta", "rho_delta_nonzero_users",
    "delta_low_to_high", "delta_nonzero_low_to_high", "delta_pair_active_low_to_high",
    "challenger_win_rate_low", "challenger_win_rate_high", "challenger_win_rate_low_to_high",
    "challenger_win_rate_pair_active_low", "challenger_win_rate_pair_active_high", "challenger_win_rate_pair_active_low_to_high",
]

for metric in contrast_metrics:
    wide = contrast_effects_by_keyset.pivot(index=contrast_index_cols, columns="keyset", values=metric).reset_index()
    for keyset in keysets:
        if keyset not in wide.columns: wide[keyset] = np.nan
    wide = wide.rename(columns={keyset: f"{metric}_k{keyset}" for keyset in keysets})
    contrast_summary = contrast_summary.merge(wide, on=contrast_index_cols, how="left", validate="one_to_one")

for metric in contrast_metrics:
    cols = [c for c in contrast_summary.columns if c.startswith(f"{metric}_k")]
    contrast_summary[f"{metric}_mean"] = contrast_summary[cols].mean(axis=1)
    contrast_summary[f"{metric}_sd"] = contrast_summary[cols].std(axis=1, ddof=0)

contrast_summary["rho_delta_sign"] = contrast_summary.apply(lambda row: sign_consistency(row, "rho_delta"), axis=1)
contrast_summary["rho_delta_nonzero_sign"] = contrast_summary.apply(lambda row: sign_consistency(row, "rho_delta_nonzero_users"), axis=1)
contrast_summary["delta_effect_sign"] = contrast_summary.apply(lambda row: sign_consistency(row, "delta_low_to_high"), axis=1)
contrast_summary["win_rate_effect_sign"] = contrast_summary.apply(lambda row: sign_consistency(row, "challenger_win_rate_low_to_high"), axis=1)
contrast_summary["abs_rho_delta_mean"] = contrast_summary["rho_delta_mean"].abs()
contrast_summary["abs_delta_low_to_high_mean"] = contrast_summary["delta_low_to_high_mean"].abs()
contrast_summary["abs_win_rate_low_to_high_mean"] = contrast_summary["challenger_win_rate_low_to_high_mean"].abs()
contrast_summary = contrast_summary.sort_values(["dataset", "abs_rho_delta_mean"], ascending=[True, False])
contrast_summary.to_csv(DATA_DIR / "regime_global_contrast_effects.csv", index=False)


# ------------------------------------------------------------------
# Console diagnostics
# ------------------------------------------------------------------

print("\nSaved:")
for filename in ["regime_method_response.csv", "regime_method_effects.csv", "regime_global_contrast_response.csv", "regime_global_contrast_effects.csv"]:
    print(DATA_DIR / filename)

print("\nStrongest split-stable relative method responses on primary datasets:")

stable = summary[summary["dataset"].isin(PRIMARY_DATASETS) & summary["rho_centered_sign"].isin(["positive", "negative"])].copy()

for dataset in PRIMARY_DATASETS:
    print(f"\n--- {dataset} ---")
    top = stable[stable["dataset"] == dataset].sort_values("abs_rho_centered_mean", ascending=False).head(12)
    print(top[["feature", "method", "rho_centered_mean", "rho_centered_sd", "centered_low_to_high_mean", "centered_low_to_high_sd", "rho_centered_sign"]].round(4).to_string(index=False))

print("\nStrongest split-stable challenger-vs-global-best regime effects:")

stable_contrasts = contrast_summary[
    contrast_summary["dataset"].isin(PRIMARY_DATASETS) &
    contrast_summary["rho_delta_sign"].isin(["positive", "negative"])
].copy()

for dataset in PRIMARY_DATASETS:
    print(f"\n--- {dataset} | global best = {global_best_by_dataset[dataset]} ---")
    top = stable_contrasts[stable_contrasts["dataset"] == dataset].sort_values("abs_rho_delta_mean", ascending=False).head(15)
    print(top[
        ["feature", "method", "global_method", "rho_delta_mean", "rho_delta_sd", "delta_low_to_high_mean",
         "challenger_win_rate_low_mean", "challenger_win_rate_high_mean", "challenger_win_rate_low_to_high_mean", "rho_delta_sign"]
    ].round(4).to_string(index=False))

print("\nInterpretation:")
print("  mean_ndcg: absolute method performance across the regime.")
print("  mean_centered_ndcg: method performance relative to the same user's 8-method average.")
print("  rho_centered_mean: whether a method becomes relatively stronger/weaker than the method panel as the regime feature increases.")
print("  centered_low_to_high_mean: relative NDCG change from the lowest occupied bin to the highest occupied bin.")
print("  mean_delta_ndcg: challenger minus dataset-level global-best nDCG; all users are retained.")
print("  rho_delta_mean: whether the challenger gains/loses relative to the global-best method as the regime feature increases.")
print("  challenger_win_rate_nonzero_users: challenger win rate among users where at least one of the 8 methods succeeds.")
print("  challenger_win_rate_pair_active: supplemental rate among users where either member of this pair succeeds.")
print("  Keysets are repeated stability checks, not independent replications.")
