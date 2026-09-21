from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

DATASETS = ["dunnhumby", "instacart", "sam", "tafeng", "valuedshopper"]
METHODS = ["recency", "frequency", "tifuknn", "fpmc", "triple2vec", "dnntsp", "cbp", "diffrec"]
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


def sign_label(values):
    values = pd.Series(values).dropna()
    if len(values) < 3:
        return "missing"
    if (values > 0).all():
        return "positive"
    if (values < 0).all():
        return "negative"
    return "mixed"


def fit_ols(df, y_col, feature_cols):
    cols = [y_col] + feature_cols
    d = df[cols].dropna().copy()

    if len(d) < len(feature_cols) + 5:
        return None

    active = [f for f in feature_cols if d[f].std(ddof=0) > EPS and d[f].nunique() > 1]
    if not active:
        return None

    X = np.column_stack([np.ones(len(d)), d[active].to_numpy(float)])
    y = d[y_col].to_numpy(float)

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "n_users": len(d),
        "r2": r2,
        "active_features": active,
        "coef": dict(zip(active, beta[1:])),
    }


def compute_vif(df, feature_cols):
    d = df[feature_cols].dropna().copy()
    rows = []

    if len(d) < len(feature_cols) + 5:
        return rows

    active = [f for f in feature_cols if d[f].std(ddof=0) > EPS and d[f].nunique() > 1]

    for feature in active:
        others = [f for f in active if f != feature]
        y = d[feature].to_numpy(float)

        if not others:
            vif = 1.0
        else:
            X = np.column_stack([np.ones(len(d)), d[others].to_numpy(float)])
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            vif = 1 / (1 - r2) if pd.notna(r2) and r2 < 1 else np.inf

        rows.append({"feature": feature, "vif": vif, "n_users": len(d)})

    return rows


print(f"{datetime.now()}: loading tables", flush=True)

performance = pd.read_csv(DATA_DIR / "performance.csv", dtype={"user": str})
features = pd.read_csv(DATA_DIR / "regime_features.csv", dtype={"user": str})

performance = performance[performance["dataset"].isin(DATASETS)].copy()
features = features[features["dataset"].isin(DATASETS)].copy()

regimes = (
    performance[["dataset", "user"]]
    .drop_duplicates()
    .merge(
        features[["dataset", "user"] + FEATURES],
        on=["dataset", "user"],
        how="left",
        validate="one_to_one",
    )
)

# Standardize each regime axis once per dataset over unique evaluated users.
# This gives coefficients in "NDCG change per 1 SD increase in the feature"
# and keeps the scale identical across keysets within a dataset.
for dataset in DATASETS:
    mask = regimes["dataset"] == dataset
    for feature in FEATURES:
        mean = regimes.loc[mask, feature].mean()
        std = regimes.loc[mask, feature].std(ddof=0)
        regimes.loc[mask, f"z_{feature}"] = (
            (regimes.loc[mask, feature] - mean) / std
            if pd.notna(std) and std > EPS
            else np.nan
        )

Z_FEATURES = [f"z_{f}" for f in FEATURES]

analysis = performance.merge(
    regimes[["dataset", "user"] + Z_FEATURES],
    on=["dataset", "user"],
    how="left",
    validate="many_to_one",
)

analysis["user_mean_ndcg"] = analysis[METHODS].mean(axis=1)
for method in METHODS:
    analysis[f"{method}_centered"] = analysis[method] - analysis["user_mean_ndcg"]


print(f"{datetime.now()}: fitting joint models", flush=True)

rows = []

for (dataset, keyset), df in analysis.groupby(["dataset", "keyset"]):
    # Generic task difficulty: which axes make everybody easier/harder?
    difficulty = fit_ols(df, "user_mean_ndcg", Z_FEATURES)
    if difficulty is not None:
        for zf, coef in difficulty["coef"].items():
            rows.append({
                "dataset": dataset,
                "keyset": keyset,
                "outcome": "difficulty",
                "method": "all_method_mean",
                "feature": zf.removeprefix("z_"),
                "n_users": difficulty["n_users"],
                "r2": difficulty["r2"],
                "coef": coef,
            })

    # Relative method response: which axes selectively favor each method?
    for method in METHODS:
        result = fit_ols(df, f"{method}_centered", Z_FEATURES)
        if result is None:
            continue

        for zf, coef in result["coef"].items():
            rows.append({
                "dataset": dataset,
                "keyset": keyset,
                "outcome": "centered",
                "method": method,
                "feature": zf.removeprefix("z_"),
                "n_users": result["n_users"],
                "r2": result["r2"],
                "coef": coef,
            })

effects = pd.DataFrame(rows)

summary = (
    effects
    .groupby(["dataset", "outcome", "method", "feature"], as_index=False)
    .agg(
        n_users=("n_users", "mean"),
        coef_mean=("coef", "mean"),
        coef_sd=("coef", "std"),
        r2_mean=("r2", "mean"),
        n_keysets=("coef", "count"),
    )
)

signs = (
    effects
    .groupby(["dataset", "outcome", "method", "feature"])["coef"]
    .apply(sign_label)
    .reset_index(name="coef_sign")
)

summary = summary.merge(
    signs,
    on=["dataset", "outcome", "method", "feature"],
    how="left",
)

summary["abs_coef_mean"] = summary["coef_mean"].abs()
summary = summary.sort_values(
    ["dataset", "outcome", "abs_coef_mean"],
    ascending=[True, True, False],
)


print(f"{datetime.now()}: computing multicollinearity diagnostics", flush=True)

vif_rows = []
for dataset in DATASETS:
    d = regimes[regimes["dataset"] == dataset]
    for row in compute_vif(d, Z_FEATURES):
        vif_rows.append({
            "dataset": dataset,
            "feature": row["feature"].removeprefix("z_"),
            "vif": row["vif"],
            "n_users_complete_case": row["n_users"],
        })

vif = pd.DataFrame(vif_rows).sort_values(["dataset", "vif"], ascending=[True, False])


effects.to_csv(DATA_DIR / "joint_regime_effects.csv", index=False)
summary.to_csv(DATA_DIR / "joint_regime_summary.csv", index=False)
vif.to_csv(DATA_DIR / "joint_regime_vif.csv", index=False)


print("\nStrongest split-stable conditional method responses")
print("(centered NDCG coefficient per +1 SD feature, all other axes held fixed)\n")

show = summary[
    (summary["outcome"] == "centered")
    & (summary["coef_sign"].isin(["positive", "negative"]))
].copy()

for dataset in DATASETS:
    print(f"\n=== {dataset} ===")
    block = (
        show[show["dataset"] == dataset]
        .sort_values("abs_coef_mean", ascending=False)
        .head(20)
    )
    print(
        block[
            ["method", "feature", "coef_mean", "coef_sd", "coef_sign", "n_users", "r2_mean"]
        ]
        .round(4)
        .to_string(index=False)
    )


print("\nGeneric task-difficulty effects")
print("(positive = higher feature value predicts higher mean NDCG across methods)\n")

difficulty_show = summary[summary["outcome"] == "difficulty"].copy()
for dataset in DATASETS:
    print(f"\n=== {dataset} ===")
    block = difficulty_show[difficulty_show["dataset"] == dataset].sort_values(
        "abs_coef_mean", ascending=False
    )
    print(
        block[
            ["feature", "coef_mean", "coef_sd", "coef_sign", "n_users", "r2_mean"]
        ]
        .round(4)
        .to_string(index=False)
    )


print("\nVIF diagnostics")
print("Rule of thumb only: >5 deserves attention; >10 is substantial multicollinearity.\n")
print(vif.round(2).to_string(index=False))

print("\nSaved:")
print(DATA_DIR / "joint_regime_effects.csv")
print(DATA_DIR / "joint_regime_summary.csv")
print(DATA_DIR / "joint_regime_vif.csv")
