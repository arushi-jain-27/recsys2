# analysis/07_bootstrap_joint_regime.py

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

DATASETS = ["dunnhumby", "instacart", "sam", "tafeng", "valuedshopper"]
METHODS = ["recency", "frequency", "tifuknn", "fpmc", "triple2vec", "dnntsp", "cbp", "diffrec"]
FEATURES = ["n_baskets", "avg_basket_size", "repeat_rate", "adjacent_basket_jaccard", "recent_basket_novelty", "popularity_profile", "neighbor_similarity", "transition_strength", "relational_strength"]

N_BOOT = 1000
CI_LEVEL = 0.95
RANDOM_SEED = 42
EPS = 1e-12

print(f"{datetime.now()}: loading inputs", flush=True)

performance = pd.read_csv(DATA_DIR / "performance.csv", dtype={"user": str})
features = pd.read_csv(DATA_DIR / "regime_features.csv", dtype={"user": str})
performance = performance[performance["dataset"].isin(DATASETS)].copy()
features = features[features["dataset"].isin(DATASETS)].copy()

assert not performance.duplicated(["dataset", "keyset", "user"]).any()
assert not features.duplicated(["dataset", "user"]).any()
assert performance[METHODS].notna().all().all()

# Match Analysis 05 exactly: standardize behavioral features once within each dataset,
# using all evaluated users across the three keysets.
regimes = performance[["dataset", "user"]].drop_duplicates().merge(
    features[["dataset", "user"] + FEATURES],
    on=["dataset", "user"], how="left", validate="one_to_one"
)

for dataset in DATASETS:
    mask = regimes["dataset"] == dataset
    for feature in FEATURES:
        mean = regimes.loc[mask, feature].mean()
        std = regimes.loc[mask, feature].std(ddof=0)
        regimes.loc[mask, f"z_{feature}"] = (
            (regimes.loc[mask, feature] - mean) / std
            if pd.notna(std) and std > EPS else np.nan
        )

Z_FEATURES = [f"z_{f}" for f in FEATURES]

analysis = performance.merge(
    regimes[["dataset", "user"] + Z_FEATURES],
    on=["dataset", "user"], how="left", validate="many_to_one"
)

analysis["user_mean_ndcg"] = analysis[METHODS].mean(axis=1)
for method in METHODS:
    analysis[f"{method}_centered"] = analysis[method] - analysis["user_mean_ndcg"]

alpha = (1 - CI_LEVEL) / 2
rng = np.random.default_rng(RANDOM_SEED)
rows = []

print(f"{datetime.now()}: bootstrapping centered joint models", flush=True)

for (dataset, keyset), df0 in analysis.groupby(["dataset", "keyset"], sort=False):
    # Same complete-case population for all eight centered outcomes because the
    # performance columns are complete and all methods use the same 9 features.
    cols = ["user"] + Z_FEATURES + [f"{m}_centered" for m in METHODS]
    df = df0[cols].dropna().reset_index(drop=True)

    active = [f for f in Z_FEATURES if df[f].std(ddof=0) > EPS and df[f].nunique() > 1]
    assert len(active) == len(Z_FEATURES), f"{dataset} keyset {keyset}: inactive feature(s)"

    X = np.column_stack([np.ones(len(df)), df[active].to_numpy(float)])
    Y = df[[f"{m}_centered" for m in METHODS]].to_numpy(float)
    n = len(df)

    # Point estimate: identical OLS specification to Analysis 05, but all
    # method outcomes are fit together because they share the same X.
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    # Shape: bootstrap x coefficient(including intercept) x method.
    boot_beta = np.empty((N_BOOT, X.shape[1], len(METHODS)), dtype=float)

    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        Xb, Yb = X[idx], Y[idx]
        bb, _, _, _ = np.linalg.lstsq(Xb, Yb, rcond=None)
        boot_beta[b] = bb

    for j, z_feature in enumerate(active, start=1):
        feature = z_feature.removeprefix("z_")
        for k, method in enumerate(METHODS):
            samples = boot_beta[:, j, k]
            coef = beta[j, k]
            ci_low, ci_high = np.quantile(samples, [alpha, 1 - alpha])

            if ci_low > 0:
                ci_sign = "positive"
            elif ci_high < 0:
                ci_sign = "negative"
            else:
                ci_sign = "overlaps_zero"

            rows.append({
                "dataset": dataset,
                "keyset": keyset,
                "method": method,
                "feature": feature,
                "n_users": n,
                "coef": coef,
                "bootstrap_mean": samples.mean(),
                "bootstrap_sd": samples.std(ddof=1),
                "ci_level": CI_LEVEL,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_excludes_zero": ci_sign != "overlaps_zero",
                "ci_sign": ci_sign,
            })

    print(f"{datetime.now()}: finished {dataset} keyset {keyset} ({n} users)", flush=True)

effects = pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Dataset-level stability summary.
#
# Keysets are repeated stability checks, NOT independent replications.
# Therefore we do not pool them into a meta-analytic p-value or CI.
#
# Evidence hierarchy:
#   strong      = same point-estimate direction in all keysets AND every
#                 keyset's bootstrap CI excludes zero in that direction
#   moderate    = same point-estimate direction in all keysets AND >=2/3
#                 keyset CIs exclude zero in that direction
#   directional = same point-estimate direction in all keysets, but <2 CIs
#                 exclude zero
#   mixed       = point-estimate directions differ across keysets
# ---------------------------------------------------------------------

summary_rows = []

for (dataset, method, feature), g in effects.groupby(["dataset", "method", "feature"], sort=False):
    coefs = g["coef"].to_numpy(float)
    n_keysets = len(g)

    all_positive = (coefs > 0).all()
    all_negative = (coefs < 0).all()

    if all_positive:
        direction = "positive"
        n_ci_support = int((g["ci_sign"] == "positive").sum())
    elif all_negative:
        direction = "negative"
        n_ci_support = int((g["ci_sign"] == "negative").sum())
    else:
        direction = "mixed"
        n_ci_support = 0

    if direction == "mixed":
        evidence = "mixed"
    elif n_ci_support == n_keysets:
        evidence = "strong"
    elif n_ci_support >= 2:
        evidence = "moderate"
    else:
        evidence = "directional"

    summary_rows.append({
        "dataset": dataset,
        "method": method,
        "feature": feature,
        "n_keysets": n_keysets,
        "coef_mean": coefs.mean(),
        "coef_sd_across_keysets": coefs.std(ddof=0),
        "point_direction": direction,
        "n_keysets_ci_support": n_ci_support,
        "evidence": evidence,
    })

summary = pd.DataFrame(summary_rows)

# Cross-dataset summary: descriptive only. Five datasets are not treated as
# samples from a dataset population and no dataset-level significance test is run.
cross_rows = []

for (method, feature), g in summary.groupby(["method", "feature"], sort=False):
    n_pos = int((g["point_direction"] == "positive").sum())
    n_neg = int((g["point_direction"] == "negative").sum())
    n_mixed = int((g["point_direction"] == "mixed").sum())

    if n_pos == len(DATASETS):
        cross_direction = "positive_5of5"
    elif n_neg == len(DATASETS):
        cross_direction = "negative_5of5"
    else:
        cross_direction = "mixed_across_datasets"

    cross_rows.append({
        "method": method,
        "feature": feature,
        "coef_mean_across_datasets": g["coef_mean"].mean(),
        "n_datasets_positive": n_pos,
        "n_datasets_negative": n_neg,
        "n_datasets_mixed_keysets": n_mixed,
        "n_datasets_strong": int((g["evidence"] == "strong").sum()),
        "n_datasets_moderate": int((g["evidence"] == "moderate").sum()),
        "n_datasets_directional": int((g["evidence"] == "directional").sum()),
        "n_datasets_mixed": int((g["evidence"] == "mixed").sum()),
        "cross_dataset_direction": cross_direction,
    })

cross = pd.DataFrame(cross_rows)

effects.to_csv(DATA_DIR / "joint_regime_bootstrap_effects.csv", index=False)
summary.to_csv(DATA_DIR / "joint_regime_bootstrap_summary.csv", index=False)
cross.to_csv(DATA_DIR / "joint_regime_bootstrap_cross_dataset.csv", index=False)

print("\n====================================================")
print("BOOTSTRAP UNCERTAINTY FOR JOINT CENTERED EFFECTS")
print("====================================================")

print("\n5/5 directionally consistent relationships:\n")
show = cross[cross["cross_dataset_direction"].isin(["positive_5of5", "negative_5of5"])].copy()
show["abs_coef"] = show["coef_mean_across_datasets"].abs()
show = show.sort_values("abs_coef", ascending=False)
print(show[[
    "method", "feature", "coef_mean_across_datasets",
    "cross_dataset_direction", "n_datasets_strong",
    "n_datasets_moderate", "n_datasets_directional",
]].round(4).to_string(index=False))

HEADLINE = [
    ("fpmc", "repeat_rate"),
    ("fpmc", "n_baskets"),
    ("recency", "repeat_rate"),
    ("recency", "adjacent_basket_jaccard"),
    ("diffrec", "avg_basket_size"),
    ("dnntsp", "recent_basket_novelty"),
    ("tifuknn", "neighbor_similarity"),
    ("fpmc", "transition_strength"),
    ("dnntsp", "transition_strength"),
    ("triple2vec", "relational_strength"),
]

print("\nHeadline / architecture-aligned relationships by dataset:\n")

for method, feature in HEADLINE:
    block = summary[(summary["method"] == method) & (summary["feature"] == feature)].copy()
    if block.empty:
        continue

    print(f"\n--- {method} × {feature} ---")
    print(block[[
        "dataset", "coef_mean", "coef_sd_across_keysets",
        "point_direction", "n_keysets_ci_support", "evidence",
    ]].round(4).to_string(index=False))

print("\nSaved:")
for filename in [
    "joint_regime_bootstrap_effects.csv",
    "joint_regime_bootstrap_summary.csv",
    "joint_regime_bootstrap_cross_dataset.csv",
]:
    print(DATA_DIR / filename)

print("\nInterpretation:")
print("  * Coefficients reproduce Analysis 05's joint centered specification.")
print("  * Predictors are standardized within dataset before keyset-specific fitting.")
print("  * A coefficient is an association with method-relative nDCG, conditional on the other 8 behavioral axes.")
print("  * Bootstrap intervals resample users within each dataset × keyset.")
print("  * The three keysets are stability checks, not independent replications.")
print("  * 'Strong/moderate/directional' summarize consistency; they are not hypothesis-test labels.")
print("  * Cross-dataset 5/5 signs are descriptive consistency, not evidence of a causal effect.")