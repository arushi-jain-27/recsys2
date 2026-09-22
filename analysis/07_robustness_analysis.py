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

N_BOOT = 500
RANDOM_SEED = 42
EPS = 1e-12

# Keep this sparse. "expected" cells should remain only if the original
# algorithm/paper clearly motivates the positive alignment.
CLAIMS = [
    {"claim_type": "expected", "method": "frequency", "feature": "repeat_rate", "expected_sign": 1},
    {"claim_type": "expected", "method": "tifuknn", "feature": "repeat_rate", "expected_sign": 1},
    {"claim_type": "expected", "method": "cbp", "feature": "repeat_rate", "expected_sign": 1},
    {"claim_type": "expected", "method": "recency", "feature": "adjacent_basket_jaccard", "expected_sign": 1},
    {"claim_type": "expected", "method": "fpmc", "feature": "transition_strength", "expected_sign": 1},
    {"claim_type": "expected", "method": "dnntsp", "feature": "transition_strength", "expected_sign": 1},
    {"claim_type": "expected", "method": "triple2vec", "feature": "relational_strength", "expected_sign": 1},

    # Discovery / observed contrasts are kept separate from theory-derived cells.
    {"claim_type": "discovery", "method": "fpmc", "feature": "n_baskets", "expected_sign": 1},
    {"claim_type": "observed_contrast", "method": "fpmc", "feature": "repeat_rate", "expected_sign": -1},
    {"claim_type": "observed_contrast", "method": "recency", "feature": "repeat_rate", "expected_sign": -1},
]


def sign_label(values):
    values = pd.Series(values).dropna()
    if len(values) < 3:
        return "missing"
    if (values > 0).all():
        return "positive"
    if (values < 0).all():
        return "negative"
    return "mixed"


def safe_spearman(x, y):
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return x.corr(y, method="spearman")


def fit_matrix_ols(X, Y):
    if len(X) <= X.shape[1]:
        return None
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


print(f"{datetime.now()}: loading inputs", flush=True)

performance = pd.read_csv(
    DATA_DIR / "performance.csv",
    dtype={"user": str},
)
features = pd.read_csv(
    DATA_DIR / "regime_features.csv",
    dtype={"user": str},
)
univariate = pd.read_csv(
    DATA_DIR / "regime_method_effects.csv",
)

performance = performance[
    performance["dataset"].isin(DATASETS)
].copy()
features = features[
    features["dataset"].isin(DATASETS)
].copy()

assert not performance.duplicated(["dataset", "keyset", "user"]).any()
assert not features.duplicated(["dataset", "user"]).any()

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

# Match Analysis 05 exactly: standardize once per dataset over unique
# evaluated users, not separately per keyset.
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
    analysis[f"{method}_centered"] = (
        analysis[method] - analysis["user_mean_ndcg"]
    )

rng = np.random.default_rng(RANDOM_SEED)
rows = []

print(f"{datetime.now()}: bootstrapping sparse claim set", flush=True)

for (dataset, keyset), df in analysis.groupby(["dataset", "keyset"]):
    complete_cols = (
        Z_FEATURES
        + METHODS
        + [f"{m}_centered" for m in METHODS]
    )
    d = df.dropna(subset=complete_cols).copy()

    if len(d) < len(FEATURES) + 10:
        continue

    X = np.column_stack([
        np.ones(len(d)),
        d[Z_FEATURES].to_numpy(float),
    ])
    Y = d[[f"{m}_centered" for m in METHODS]].to_numpy(float)

    beta_full = fit_matrix_ols(X, Y)
    if beta_full is None:
        continue

    method_to_col = {
        method: idx
        for idx, method in enumerate(METHODS)
    }
    feature_to_row = {
        feature: idx + 1
        for idx, feature in enumerate(FEATURES)
    }

    # All-zero-user sensitivity.
    nonzero_mask = ~(
        d[METHODS].fillna(0.0).eq(0.0).all(axis=1)
    ).to_numpy()

    beta_nonzero = fit_matrix_ols(
        X[nonzero_mask],
        Y[nonzero_mask],
    )

    boot_values = {
        (c["method"], c["feature"]): []
        for c in CLAIMS
    }

    for _ in range(N_BOOT):
        sample_idx = rng.integers(
            0,
            len(d),
            size=len(d),
        )

        beta_boot = fit_matrix_ols(
            X[sample_idx],
            Y[sample_idx],
        )
        if beta_boot is None:
            continue

        for claim in CLAIMS:
            method = claim["method"]
            feature = claim["feature"]
            boot_values[(method, feature)].append(
                beta_boot[
                    feature_to_row[feature],
                    method_to_col[method],
                ]
            )

    for claim in CLAIMS:
        method = claim["method"]
        feature = claim["feature"]
        expected_sign = claim["expected_sign"]

        coef_full = beta_full[
            feature_to_row[feature],
            method_to_col[method],
        ]

        coef_nonzero = (
            beta_nonzero[
                feature_to_row[feature],
                method_to_col[method],
            ]
            if beta_nonzero is not None
            else np.nan
        )

        boot = np.asarray(
            boot_values[(method, feature)],
            dtype=float,
        )

        if len(boot):
            ci_low, ci_high = np.quantile(
                boot,
                [0.025, 0.975],
            )
            boot_median = np.median(boot)
        else:
            ci_low = np.nan
            ci_high = np.nan
            boot_median = np.nan

        raw_rho = safe_spearman(
            d[f"z_{feature}"],
            d[method],
        )
        centered_rho = safe_spearman(
            d[f"z_{feature}"],
            d[f"{method}_centered"],
        )

        rows.append({
            "dataset": dataset,
            "keyset": keyset,
            "claim_type": claim["claim_type"],
            "method": method,
            "feature": feature,
            "expected_sign": expected_sign,
            "n_users_complete": len(d),
            "n_users_nonzero": int(nonzero_mask.sum()),
            "all_zero_fraction": 1.0 - nonzero_mask.mean(),
            "coef_full": coef_full,
            "bootstrap_median": boot_median,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_excludes_zero": (
                (ci_low > 0) or (ci_high < 0)
                if pd.notna(ci_low) and pd.notna(ci_high)
                else False
            ),
            "matches_expected_sign": (
                np.sign(coef_full) == expected_sign
                if coef_full != 0
                else False
            ),
            "coef_nonzero_users": coef_nonzero,
            "nonzero_same_sign": (
                np.sign(coef_full) == np.sign(coef_nonzero)
                if pd.notna(coef_nonzero)
                and coef_full != 0
                and coef_nonzero != 0
                else False
            ),
            "rho_raw": raw_rho,
            "rho_centered": centered_rho,
        })

bootstrap = pd.DataFrame(rows)

summary = (
    bootstrap
    .groupby(
        ["dataset", "claim_type", "method", "feature", "expected_sign"],
        as_index=False,
    )
    .agg(
        n_users_complete=("n_users_complete", "mean"),
        all_zero_fraction=("all_zero_fraction", "mean"),
        coef_mean=("coef_full", "mean"),
        coef_sd=("coef_full", "std"),
        coef_nonzero_mean=("coef_nonzero_users", "mean"),
        rho_raw_mean=("rho_raw", "mean"),
        rho_centered_mean=("rho_centered", "mean"),
        n_keysets=("keyset", "nunique"),
        n_keysets_ci_excludes_zero=("ci_excludes_zero", "sum"),
        n_keysets_matches_expected_sign=("matches_expected_sign", "sum"),
        n_keysets_nonzero_same_sign=("nonzero_same_sign", "sum"),
    )
)

coef_signs = (
    bootstrap
    .groupby(
        ["dataset", "claim_type", "method", "feature", "expected_sign"]
    )["coef_full"]
    .apply(sign_label)
    .reset_index(name="conditional_sign_across_keysets")
)

nonzero_signs = (
    bootstrap
    .groupby(
        ["dataset", "claim_type", "method", "feature", "expected_sign"]
    )["coef_nonzero_users"]
    .apply(sign_label)
    .reset_index(name="nonzero_sign_across_keysets")
)

summary = (
    summary
    .merge(
        coef_signs,
        on=["dataset", "claim_type", "method", "feature", "expected_sign"],
        how="left",
    )
    .merge(
        nonzero_signs,
        on=["dataset", "claim_type", "method", "feature", "expected_sign"],
        how="left",
    )
)

# Pull in Analysis 04's low-to-high effects for easy raw-vs-centered reading.
uni_keep = univariate[
    [
        "dataset",
        "feature",
        "method",
        "rho_raw_mean",
        "rho_centered_mean",
        "raw_low_to_high_mean",
        "centered_low_to_high_mean",
        "rho_raw_sign",
        "rho_centered_sign",
        "centered_effect_sign",
    ]
].copy()

uni_keep = uni_keep.rename(columns={
    "rho_raw_mean": "analysis04_rho_raw_mean",
    "rho_centered_mean": "analysis04_rho_centered_mean",
    "raw_low_to_high_mean": "analysis04_raw_low_to_high_mean",
    "centered_low_to_high_mean": "analysis04_centered_low_to_high_mean",
    "rho_raw_sign": "analysis04_rho_raw_sign",
    "rho_centered_sign": "analysis04_rho_centered_sign",
    "centered_effect_sign": "analysis04_centered_effect_sign",
})

summary = summary.merge(
    uni_keep,
    on=["dataset", "feature", "method"],
    how="left",
    validate="one_to_one",
)

summary["abs_coef_mean"] = summary["coef_mean"].abs()

bootstrap.to_csv(
    DATA_DIR / "robustness_bootstrap_by_keyset.csv",
    index=False,
)
summary.to_csv(
    DATA_DIR / "robustness_claim_summary.csv",
    index=False,
)

print("\nSaved:")
print(DATA_DIR / "robustness_bootstrap_by_keyset.csv")
print(DATA_DIR / "robustness_claim_summary.csv")

print("\nClaim-level robustness summary:")
print(
    summary[
        [
            "dataset",
            "claim_type",
            "method",
            "feature",
            "coef_mean",
            "conditional_sign_across_keysets",
            "n_keysets_ci_excludes_zero",
            "n_keysets_matches_expected_sign",
            "all_zero_fraction",
            "coef_nonzero_mean",
            "nonzero_sign_across_keysets",
            "analysis04_rho_raw_mean",
            "analysis04_rho_centered_mean",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\nInterpretation:")
print("  bootstrap CIs are within dataset x keyset; the 3 keysets are not independent replications.")
print("  all-zero sensitivity checks whether the conditional sign survives when universally missed users are removed.")
print("  raw vs centered correlations distinguish generic task easiness from relative method advantage.")
print("  a CI containing zero is not proof of a null/equivalence; describe such expected cells as weak or inconclusive.")
