from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

DATASETS = ["dunnhumby", "instacart", "sam", "tafeng", "valuedshopper"]
METHODS = ["recency", "frequency", "tifuknn", "fpmc", "triple2vec", "dnntsp", "cbp", "diffrec"]
FEATURES = ["n_baskets", "avg_basket_size", "repeat_rate", "adjacent_basket_jaccard", "recent_basket_novelty", "popularity_profile", "neighbor_similarity", "transition_strength", "relational_strength"]
EPS = 1e-12

def sign_label(values):
    v = pd.Series(values).dropna()
    if len(v) < 3: return "missing"
    if (v > 0).all(): return "positive"
    if (v < 0).all(): return "negative"
    return "mixed"

def fit_ols(df, y_col, feature_cols):
    d = df[[y_col] + feature_cols].dropna()
    if len(d) < len(feature_cols) + 5: return None
    active = [f for f in feature_cols if d[f].std(ddof=0) > EPS and d[f].nunique() > 1]
    if not active: return None
    X = np.column_stack([np.ones(len(d)), d[active].to_numpy(float)])
    y = d[y_col].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res, ss_tot = np.sum((y - pred) ** 2), np.sum((y - y.mean()) ** 2)
    return {"n_users": len(d), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, "coef": dict(zip(active, beta[1:]))}

def compute_vif(df, feature_cols):
    d = df[feature_cols].dropna()
    if len(d) < len(feature_cols) + 5: return []
    active = [f for f in feature_cols if d[f].std(ddof=0) > EPS and d[f].nunique() > 1]
    rows = []
    for feature in active:
        others, y = [f for f in active if f != feature], d[feature].to_numpy(float)
        if not others: vif = 1.0
        else:
            X = np.column_stack([np.ones(len(d)), d[others].to_numpy(float)])
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            ss_res, ss_tot = np.sum((y - pred) ** 2), np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            vif = 1 / (1 - r2) if pd.notna(r2) and r2 < 1 else np.inf
        rows.append({"feature": feature, "vif": vif, "n_users": len(d)})
    return rows

def append_fit(rows, df, y_col, dataset, keyset, outcome, method, reference_method, population):
    result = fit_ols(df, y_col, Z_FEATURES)
    if result is None: return
    for zf, coef in result["coef"].items():
        rows.append({"dataset": dataset, "keyset": keyset, "outcome": outcome, "population": population, "method": method, "reference_method": reference_method, "feature": zf.removeprefix("z_"), "n_users": result["n_users"], "r2": result["r2"], "coef": coef})

print(f"{datetime.now()}: loading inputs", flush=True)

performance = pd.read_csv(DATA_DIR / "performance.csv", dtype={"user": str})
features = pd.read_csv(DATA_DIR / "regime_features.csv", dtype={"user": str})
performance = performance[performance["dataset"].isin(DATASETS)].copy()
features = features[features["dataset"].isin(DATASETS)].copy()

assert not performance.duplicated(["dataset", "keyset", "user"]).any()
assert not features.duplicated(["dataset", "user"]).any()

regimes = performance[["dataset", "user"]].drop_duplicates().merge(features[["dataset", "user"] + FEATURES], on=["dataset", "user"], how="left", validate="one_to_one")

for dataset in DATASETS:
    mask = regimes["dataset"] == dataset
    for feature in FEATURES:
        mean, std = regimes.loc[mask, feature].mean(), regimes.loc[mask, feature].std(ddof=0)
        regimes.loc[mask, f"z_{feature}"] = (regimes.loc[mask, feature] - mean) / std if pd.notna(std) and std > EPS else np.nan

Z_FEATURES = [f"z_{f}" for f in FEATURES]
analysis = performance.merge(regimes[["dataset", "user"] + Z_FEATURES], on=["dataset", "user"], how="left", validate="many_to_one")
analysis["user_mean_ndcg"] = analysis[METHODS].mean(axis=1)
for method in METHODS: analysis[f"{method}_centered"] = analysis[method] - analysis["user_mean_ndcg"]

# Same descriptive global reference as Analysis 04/08: mean within keyset, then equal-weight keyset means.
global_method_means = performance.groupby(["dataset", "keyset"])[METHODS].mean().groupby("dataset").mean()
global_best_by_dataset = global_method_means.idxmax(axis=1).to_dict()

print("\nGlobal reference method by dataset:")
for dataset, method in global_best_by_dataset.items(): print(f"  {dataset}: {method}")

print(f"\n{datetime.now()}: fitting joint models", flush=True)
rows = []

for (dataset, keyset), df0 in analysis.groupby(["dataset", "keyset"]):
    df = df0.copy()
    global_method = global_best_by_dataset[dataset]
    nonzero = df[METHODS].max(axis=1) > EPS

    append_fit(rows, df, "user_mean_ndcg", dataset, keyset, "difficulty", "all_method_mean", "", "all")

    for method in METHODS:
        append_fit(rows, df, f"{method}_centered", dataset, keyset, "centered", method, "method_panel_mean", "all")

    for method in METHODS:
        if method == global_method: continue
        y_col = f"{method}_minus_{global_method}"
        df[y_col] = df[method] - df[global_method]
        append_fit(rows, df, y_col, dataset, keyset, "global_delta", method, global_method, "all")
        append_fit(rows, df.loc[nonzero].copy(), y_col, dataset, keyset, "global_delta", method, global_method, "nonzero_any_method")

effects = pd.DataFrame(rows)
group_cols = ["dataset", "outcome", "population", "method", "reference_method", "feature"]

summary = effects.groupby(group_cols, as_index=False).agg(
    n_users=("n_users", "mean"), coef_mean=("coef", "mean"), coef_sd=("coef", lambda x: x.std(ddof=0)),
    r2_mean=("r2", "mean"), n_keysets=("coef", "count")
)
signs = effects.groupby(group_cols)["coef"].apply(sign_label).reset_index(name="coef_sign")
summary = summary.merge(signs, on=group_cols, how="left")
summary["abs_coef_mean"] = summary["coef_mean"].abs()
summary = summary.sort_values(["dataset", "outcome", "population", "abs_coef_mean"], ascending=[True, True, True, False])

print(f"{datetime.now()}: computing VIF diagnostics", flush=True)
vif_rows = []
for dataset in DATASETS:
    for row in compute_vif(regimes[regimes["dataset"] == dataset], Z_FEATURES):
        vif_rows.append({"dataset": dataset, "feature": row["feature"].removeprefix("z_"), "vif": row["vif"], "n_users_complete_case": row["n_users"]})
vif = pd.DataFrame(vif_rows).sort_values(["dataset", "vif"], ascending=[True, False])

effects.to_csv(DATA_DIR / "joint_regime_effects.csv", index=False)
summary.to_csv(DATA_DIR / "joint_regime_summary.csv", index=False)
vif.to_csv(DATA_DIR / "joint_regime_vif.csv", index=False)

print("\nStrongest split-stable challenger-vs-global conditional effects")
print("(all users; coefficient = change in challenger-minus-global nDCG per +1 SD feature, other axes held fixed)\n")
show = summary[(summary["outcome"] == "global_delta") & (summary["population"] == "all") & summary["coef_sign"].isin(["positive", "negative"])].copy()
for dataset in DATASETS:
    print(f"\n=== {dataset} | global = {global_best_by_dataset[dataset]} ===")
    block = show[show["dataset"] == dataset].sort_values("abs_coef_mean", ascending=False).head(20)
    print(block[["method", "feature", "coef_mean", "coef_sd", "coef_sign", "n_users", "r2_mean"]].round(4).to_string(index=False))

print("\nAll-user vs nonzero-user sensitivity for strongest global contrasts\n")
all_delta = summary[(summary["outcome"] == "global_delta") & (summary["population"] == "all")][["dataset", "method", "reference_method", "feature", "coef_mean", "coef_sign"]].rename(columns={"coef_mean": "coef_all", "coef_sign": "sign_all"})
nz_delta = summary[(summary["outcome"] == "global_delta") & (summary["population"] == "nonzero_any_method")][["dataset", "method", "reference_method", "feature", "coef_mean", "coef_sign"]].rename(columns={"coef_mean": "coef_nonzero", "coef_sign": "sign_nonzero"})
sensitivity = all_delta.merge(nz_delta, on=["dataset", "method", "reference_method", "feature"], how="left", validate="one_to_one")
sensitivity["abs_coef_all"] = sensitivity["coef_all"].abs()
for dataset in DATASETS:
    print(f"\n=== {dataset} ===")
    block = sensitivity[sensitivity["dataset"] == dataset].sort_values("abs_coef_all", ascending=False).head(12)
    print(block[["method", "feature", "coef_all", "sign_all", "coef_nonzero", "sign_nonzero"]].round(4).to_string(index=False))

print("\nStrongest split-stable centered method responses\n")
show_centered = summary[(summary["outcome"] == "centered") & (summary["population"] == "all") & summary["coef_sign"].isin(["positive", "negative"])].copy()
for dataset in DATASETS:
    print(f"\n=== {dataset} ===")
    block = show_centered[show_centered["dataset"] == dataset].sort_values("abs_coef_mean", ascending=False).head(12)
    print(block[["method", "feature", "coef_mean", "coef_sd", "coef_sign", "r2_mean"]].round(4).to_string(index=False))

print("\nGeneric task-difficulty effects\n")
difficulty = summary[(summary["outcome"] == "difficulty") & (summary["population"] == "all")].copy()
for dataset in DATASETS:
    print(f"\n=== {dataset} ===")
    block = difficulty[difficulty["dataset"] == dataset].sort_values("abs_coef_mean", ascending=False)
    print(block[["feature", "coef_mean", "coef_sd", "coef_sign", "r2_mean"]].round(4).to_string(index=False))

print("\nVIF diagnostics (rule of thumb only: >5 deserves attention; >10 substantial)\n")
print(vif.round(2).to_string(index=False))

print("\nSaved:")
for filename in ["joint_regime_effects.csv", "joint_regime_summary.csv", "joint_regime_vif.csv"]: print(DATA_DIR / filename)

print("\nInterpretation:")
print("  difficulty: which history axes make recommendation easier/harder for the whole method panel.")
print("  centered: method nDCG relative to the same user's 8-method mean.")
print("  global_delta: challenger nDCG minus the dataset-level global-best method's nDCG.")
print("  population=all is primary; nonzero_any_method removes users missed by all 8 methods as a sensitivity check.")
print("  coefficients use dataset-standardized history features and are conditional on all 9 axes.")
print("  the global reference is descriptive/hindsight, not a deployable selector baseline.")
print("  the 3 keysets are repeated stability checks, not independent replications.")
