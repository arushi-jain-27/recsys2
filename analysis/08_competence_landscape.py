# analysis/08_competence_landscape.py

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

METHODS = [
    "recency", "frequency", "tifuknn", "fpmc",
    "triple2vec", "dnntsp", "cbp", "diffrec",
]

EPS = 1e-12
TIE_ATOL = 1e-12


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if len(x) else np.nan


def safe_median(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(x)) if len(x) else np.nan


def safe_quantile(x, q):
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, q)) if len(x) else np.nan


def share(count, denominator):
    return float(count / denominator) if denominator else np.nan


performance = pd.read_csv(DATA_DIR / "performance.csv", dtype={"user": str})

assert not performance.duplicated(["dataset", "keyset", "user"]).any()
assert performance[METHODS].notna().all().all(), "Missing method scores found in performance.csv"

summary_rows = []
winner_rows = []
pairwise_rows = []
user_parts = []

# Analyze each dataset x keyset independently
for (dataset, keyset), df in performance.groupby(["dataset", "keyset"], sort=False):
    df = df.copy().reset_index(drop=True)
    scores = df[METHODS].to_numpy(float)
    n_users = len(df)

    # Global single-best method. If aggregate means tie, take the first method in METHODS.
    method_means = scores.mean(axis=0)
    best_mean = method_means.max()
    global_best_mask = np.isclose(method_means, best_mean, atol=TIE_ATOL, rtol=0)
    global_best_indices = np.flatnonzero(global_best_mask)
    global_best_methods = [METHODS[j] for j in global_best_indices]
    global_idx = int(global_best_indices[0])
    global_method = METHODS[global_idx]
    global_user_scores = scores[:, global_idx]

    # User-level oracle. All-zero users do not count as won by every method.
    oracle_scores = scores.max(axis=1)
    all_zero = np.max(np.abs(scores), axis=1) <= EPS
    winner_matrix = np.isclose(scores, oracle_scores[:, None], atol=TIE_ATOL, rtol=0)
    winner_matrix[all_zero, :] = False
    n_winners = winner_matrix.sum(axis=1)
    # Fractional credit: 1 winner -> 1, 2-way tie -> 0.5 each, etc.
    winner_credit = np.divide(
        winner_matrix.astype(float),
        n_winners[:, None],
        out=np.zeros_like(scores, dtype=float),
        where=n_winners[:, None] > 0,
    )
    nonzero = ~all_zero
    n_nonzero = int(nonzero.sum())

    # Regret of deploying the global single-best method
    global_regret = oracle_scores - global_user_scores
    global_regret[np.abs(global_regret) <= EPS] = 0.0

    # How decisive is the per-user winner?
    sorted_scores = np.sort(scores, axis=1)
    second_best_scores = sorted_scores[:, -2]
    oracle_margin = oracle_scores - second_best_scores
    oracle_margin[np.abs(oracle_margin) <= EPS] = 0.0

    oracle_winner_strings = []
    unique_winner_strings = []
    for i in range(n_users):
        if all_zero[i]:
            oracle_winner_strings.append("")
            unique_winner_strings.append("")
            continue
        winners = [METHODS[j] for j in np.flatnonzero(winner_matrix[i])]
        oracle_winner_strings.append(";".join(winners))
        unique_winner_strings.append(winners[0] if len(winners) == 1 else "")

    user_df = df[["dataset", "keyset", "user"]].copy()
    user_df["global_best_method"] = global_method
    user_df["global_best_score"] = global_user_scores
    user_df["oracle_score"] = oracle_scores
    user_df["global_regret"] = global_regret
    user_df["all_methods_zero"] = all_zero
    user_df["n_oracle_winners"] = n_winners
    user_df["oracle_winners"] = oracle_winner_strings
    user_df["unique_oracle_winner"] = unique_winner_strings
    user_df["second_best_score"] = second_best_scores
    user_df["oracle_margin"] = oracle_margin
    user_parts.append(user_df)

    global_is_oracle = winner_matrix[:, global_idx]
    global_mean = global_user_scores.mean()
    oracle_mean = oracle_scores.mean()
    headroom = oracle_mean - global_mean
    nonzero_global = safe_mean(global_user_scores[nonzero])
    nonzero_oracle = safe_mean(oracle_scores[nonzero])

    summary_rows.append({
        "dataset": dataset,
        "keyset": keyset,
        "n_users": n_users,
        "n_nonzero_users": n_nonzero,
        "n_all_zero_users": int(all_zero.sum()),
        "all_zero_rate": all_zero.mean(),
        "global_best_method": global_method,
        "global_best_methods_if_tied": ";".join(global_best_methods),
        "n_global_best_ties": len(global_best_methods),
        "global_best_mean_ndcg": global_mean,
        "oracle_mean_ndcg": oracle_mean,
        "oracle_headroom_abs": headroom,
        "oracle_headroom_relative": headroom / global_mean if global_mean > EPS else np.nan,
        "global_best_mean_ndcg_nonzero_users": nonzero_global,
        "oracle_mean_ndcg_nonzero_users": nonzero_oracle,
        "oracle_headroom_abs_nonzero_users": (
            nonzero_oracle - nonzero_global if n_nonzero else np.nan
        ),
        "mean_global_regret": global_regret.mean(),
        "median_global_regret": np.median(global_regret),
        "p75_global_regret": np.quantile(global_regret, 0.75),
        "p90_global_regret": np.quantile(global_regret, 0.90),
        "mean_global_regret_nonzero_users": safe_mean(global_regret[nonzero]),
        "fraction_users_with_positive_global_regret": (global_regret > EPS).mean(),
        "global_method_oracle_share_all_users": global_is_oracle.mean(),
        "global_method_oracle_share_nonzero_users": share(global_is_oracle[nonzero].sum(), n_nonzero),
        "unique_winner_rate_nonzero_users": share(((n_winners == 1) & nonzero).sum(), n_nonzero),
        "multiwinner_tie_rate_nonzero_users": share(((n_winners > 1) & nonzero).sum(), n_nonzero),
        "mean_oracle_margin_nonzero_users": safe_mean(oracle_margin[nonzero]),
        "median_oracle_margin_nonzero_users": safe_median(oracle_margin[nonzero]),
    })

    ranks = (
        pd.Series(method_means).rank(method="min", ascending=False).astype(int).to_numpy()
    )
    for j, method in enumerate(METHODS):
        fractional_credit = winner_credit[:, j]
        unique_win = winner_matrix[:, j] & (n_winners == 1)
        credit_sum = fractional_credit.sum()
        winner_rows.append({
            "dataset": dataset,
            "keyset": keyset,
            "method": method,
            "mean_ndcg": method_means[j],
            "global_rank": int(ranks[j]),
            "is_global_best": bool(global_best_mask[j]),
            "fractional_winner_credit": credit_sum,
            "winner_share_all_users": credit_sum / n_users,
            "winner_share_nonzero_users": credit_sum / n_nonzero if n_nonzero else np.nan,
            "unique_winner_count": int(unique_win.sum()),
            "unique_winner_share_all_users": unique_win.mean(),
            "unique_winner_share_nonzero_users": share(unique_win[nonzero].sum(), n_nonzero),
            "oracle_winner_presence_share_nonzero_users": share(
                winner_matrix[nonzero, j].sum(), n_nonzero
            ),
        })

    for method_a, method_b in combinations(METHODS, 2):
        a = df[method_a].to_numpy(float)
        b = df[method_b].to_numpy(float)
        a_positive = a > EPS
        b_positive = b > EPS
        both_zero = ~a_positive & ~b_positive
        only_a = a_positive & ~b_positive
        only_b = ~a_positive & b_positive
        both_positive = a_positive & b_positive
        diff = a - b
        a_better = diff > TIE_ATOL
        b_better = diff < -TIE_ATOL
        tied = ~(a_better | b_better)
        success_union = a_positive | b_positive
        n_success_union = int(success_union.sum())
        pair_oracle = np.maximum(a, b)
        better_global_mean = max(a.mean(), b.mean())
        pair_oracle_mean = pair_oracle.mean()
        pairwise_rows.append({
            "dataset": dataset,
            "keyset": keyset,
            "method_a": method_a,
            "method_b": method_b,
            "n_users": n_users,
            "mean_a": a.mean(),
            "mean_b": b.mean(),
            "mean_difference_a_minus_b": diff.mean(),
            "mean_absolute_user_difference": np.abs(diff).mean(),
            "n_both_zero": int(both_zero.sum()),
            "n_only_a_positive": int(only_a.sum()),
            "n_only_b_positive": int(only_b.sum()),
            "n_both_positive": int(both_positive.sum()),
            "share_both_zero": both_zero.mean(),
            "share_only_a_positive": only_a.mean(),
            "share_only_b_positive": only_b.mean(),
            "share_both_positive": both_positive.mean(),
            "success_set_jaccard": (
                both_positive.sum() / n_success_union if n_success_union else np.nan
            ),
            "unique_success_fraction_of_union": (
                (only_a.sum() + only_b.sum()) / n_success_union if n_success_union else np.nan
            ),
            "n_a_better": int(a_better.sum()),
            "n_b_better": int(b_better.sum()),
            "n_tied": int(tied.sum()),
            "share_a_better": a_better.mean(),
            "share_b_better": b_better.mean(),
            "share_tied": tied.mean(),
            "pair_oracle_mean_ndcg": pair_oracle_mean,
            "better_global_method_mean_ndcg": better_global_mean,
            "pair_oracle_gain_over_better_global": pair_oracle_mean - better_global_mean,
        })

competence_user_level = pd.concat(user_parts, ignore_index=True)
competence_summary = pd.DataFrame(summary_rows)
competence_winner_share = pd.DataFrame(winner_rows)
competence_pairwise = pd.DataFrame(pairwise_rows)

competence_user_level.to_csv(DATA_DIR / "competence_user_level.csv", index=False)
competence_summary.to_csv(DATA_DIR / "competence_summary.csv", index=False)
competence_winner_share.to_csv(DATA_DIR / "competence_winner_share.csv", index=False)
competence_pairwise.to_csv(DATA_DIR / "competence_pairwise.csv", index=False)

print("\n====================================================")
print("USER-LEVEL COMPETENCE LANDSCAPE")
print("====================================================")

print("\nGlobal single-best vs oracle by dataset/keyset:\n")
show_cols = [
    "dataset", "keyset", "global_best_method", "global_best_mean_ndcg",
    "oracle_mean_ndcg", "oracle_headroom_abs", "oracle_headroom_relative",
    "all_zero_rate", "global_method_oracle_share_nonzero_users",
]
print(competence_summary[show_cols].round(4).to_string(index=False))

print("\nMean across keysets (descriptive stability only):\n")
dataset_summary = (
    competence_summary
    .groupby("dataset")
    .agg(
        global_best_mean_ndcg=("global_best_mean_ndcg", "mean"),
        oracle_mean_ndcg=("oracle_mean_ndcg", "mean"),
        oracle_headroom_abs=("oracle_headroom_abs", "mean"),
        oracle_headroom_relative=("oracle_headroom_relative", "mean"),
        all_zero_rate=("all_zero_rate", "mean"),
        global_method_oracle_share=("global_method_oracle_share_nonzero_users", "mean"),
        unique_winner_rate=("unique_winner_rate_nonzero_users", "mean"),
        multiwinner_tie_rate=("multiwinner_tie_rate_nonzero_users", "mean"),
    )
    .reset_index()
)
print(dataset_summary.round(4).to_string(index=False))
print(
    "\nNOTE: the three keysets are repeated stability checks, "
    "not independent replications."
)

print("\nFractional winner shares averaged across keysets:\n")
winner_show = (
    competence_winner_share
    .groupby(["dataset", "method"], as_index=False)
    .agg(
        mean_ndcg=("mean_ndcg", "mean"),
        winner_share=("winner_share_nonzero_users", "mean"),
        unique_winner_share=("unique_winner_share_nonzero_users", "mean"),
    )
)
winner_cols = ["method", "mean_ndcg", "winner_share", "unique_winner_share"]
for dataset in winner_show["dataset"].unique():
    print(f"\n--- {dataset} ---")
    block = winner_show[winner_show["dataset"] == dataset].sort_values(
        "winner_share", ascending=False
    )
    print(block[winner_cols].round(4).to_string(index=False))

print("\nMost complementary method pairs:\n")
pair_show = (
    competence_pairwise
    .groupby(["dataset", "method_a", "method_b"], as_index=False)
    .agg(
        pair_oracle_gain=("pair_oracle_gain_over_better_global", "mean"),
        success_set_jaccard=("success_set_jaccard", "mean"),
        unique_success_fraction=("unique_success_fraction_of_union", "mean"),
    )
)
pair_cols = [
    "method_a", "method_b", "pair_oracle_gain",
    "success_set_jaccard", "unique_success_fraction",
]
for dataset in pair_show["dataset"].unique():
    print(f"\n--- {dataset} ---")
    top = (
        pair_show[pair_show["dataset"] == dataset]
        .sort_values("pair_oracle_gain", ascending=False)
        .head(10)
    )
    print(top[pair_cols].round(4).to_string(index=False))

print("\nSaved:")
for filename in [
    "competence_user_level.csv",
    "competence_summary.csv",
    "competence_winner_share.csv",
    "competence_pairwise.csv",
]:
    print(DATA_DIR / filename)
