from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime

import json
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "dunnhumby",
    "instacart",
    "retailrocket",
    "sam",
    "taobao",
    "valuedshopper",
    "tmall",
    "tafeng",
]

TOP_NEIGHBORS = 20

# Shrink sparse estimates toward "no special signal".
#
# support = 5 -> empirical estimate receives 50% weight.
SEQUENTIAL_SHRINKAGE = 5.0
RELATIONAL_SHRINKAGE = 5.0


# ============================================================
# Loading
# ============================================================

def load_histories(dataset):
    """
    Load observed histories only.

    history.json stores leading/trailing [-1] markers, so
    baskets[1:-1] contains the observed user history used for
    prediction.

    Each basket is treated as a set: within-basket quantities
    are ignored.
    """
    path = ROOT / "datasets" / dataset / "history.json"

    with open(path) as f:
        history = json.load(f)

    histories = {
        user: [set(basket) for basket in baskets[1:-1]]
        for user, baskets in history.items()
    }

    return histories


# ============================================================
# Basic population statistics
# ============================================================

def build_basket_statistics(histories):
    """
    Basket-level item and pair occurrence counts.

    item_basket_counts[i]:
        number of baskets containing item i

    pair_counts[(i, j)]:
        number of baskets containing both i and j
    """
    item_basket_counts = Counter()
    pair_counts = Counter()

    total_baskets = 0

    for baskets in histories.values():
        for basket in baskets:
            if not basket:
                continue

            total_baskets += 1

            for item in basket:
                item_basket_counts[item] += 1

            for i, j in combinations(sorted(basket), 2):
                pair_counts[(i, j)] += 1

    return item_basket_counts, pair_counts, total_baskets


def build_popularity_percentiles(item_basket_counts):
    """
    Global item popularity percentile based on the number of
    historical baskets containing each item.

    Values closer to 1 correspond to more popular items.
    """
    df = pd.DataFrame({
        "item": list(item_basket_counts.keys()),
        "count": list(item_basket_counts.values()),
    })

    df["percentile"] = df["count"].rank(
        pct=True,
        method="average",
    )

    return dict(zip(df["item"], df["percentile"]))


# ============================================================
# Relational structure
# ============================================================

def build_pair_associations(
    item_basket_counts,
    pair_counts,
    total_baskets,
):
    """
    Build popularity-normalized positive item-pair associations.

    For each pair:

        PMI(i,j) =
            log( P(i,j) / (P(i) P(j)) )

        NPMI(i,j) =
            PMI(i,j) / -log(P(i,j))

    Then shrink rare pairs:

        weight = count(i,j) / (count(i,j) + lambda)

        association =
            weight * max(0, NPMI(i,j))

    Why:
    ----
    Raw co-occurrence counts are strongly confounded by item
    popularity. NPMI asks whether two items co-occur MORE often
    than expected given their individual popularity.

    Negative associations are set to zero because they do not
    constitute positive relational evidence available to an
    NBR model.
    """
    associations = {}

    if total_baskets == 0:
        return associations

    for (i, j), pair_count in pair_counts.items():

        p_i = item_basket_counts[i] / total_baskets
        p_j = item_basket_counts[j] / total_baskets
        p_ij = pair_count / total_baskets

        if p_i <= 0 or p_j <= 0 or p_ij <= 0:
            associations[(i, j)] = 0.0
            continue

        pmi = np.log(
            p_ij / (p_i * p_j)
        )

        denominator = -np.log(p_ij)

        # If p_ij == 1, denominator is zero.
        # In this case PMI is also zero when both items always
        # occur, so there is no meaningful positive association.
        if denominator <= 0:
            npmi = 0.0
        else:
            npmi = pmi / denominator

        positive_npmi = max(0.0, npmi)

        shrinkage_weight = (
            pair_count
            / (pair_count + RELATIONAL_SHRINKAGE)
        )

        associations[(i, j)] = (
            shrinkage_weight * positive_npmi
        )

    return associations


def compute_relational_strength(
    baskets,
    pair_associations,
):
    """
    Mean within-basket relational strength.

    First compute the mean shrunken positive NPMI for each
    basket, then average equally across baskets.

    Giving baskets equal weight prevents large baskets from
    dominating simply because they contain many more pairs.

    Singleton baskets receive relational strength 0.
    """
    if not baskets:
        return np.nan

    basket_scores = []

    for basket in baskets:

        if len(basket) < 2:
            basket_scores.append(0.0)
            continue

        scores = []

        for i, j in combinations(sorted(basket), 2):
            scores.append(
                pair_associations.get((i, j), 0.0)
            )

        basket_scores.append(
            np.mean(scores) if scores else 0.0
        )

    return float(np.mean(basket_scores))


# ============================================================
# Sequential structure
# ============================================================

def build_transition_statistics(histories):
    """
    Estimate:

        Q(j) = global distribution of next-basket items

    and:

        P(j | i) = successor distribution given that item i
                   appeared in the current basket

    Every successor basket contributes TOTAL mass 1.

    Therefore a 20-item next basket does not contribute 20x
    as much probability mass as a singleton next basket.
    """
    global_next_mass = Counter()

    successor_mass = defaultdict(Counter)
    origin_support = Counter()

    total_transitions = 0

    for baskets in histories.values():

        for t in range(len(baskets) - 1):

            current = baskets[t]
            nxt = baskets[t + 1]

            if not current or not nxt:
                continue

            total_transitions += 1

            next_item_mass = 1.0 / len(nxt)

            # Global next-item distribution
            for j in nxt:
                global_next_mass[j] += next_item_mass

            # Item-conditioned successor distributions
            for i in current:

                origin_support[i] += 1

                for j in nxt:
                    successor_mass[i][j] += next_item_mass

    if total_transitions == 0:
        return {}, {}, {}

    global_next_prob = {
        item: mass / total_transitions
        for item, mass in global_next_mass.items()
    }

    conditional_probs = {}

    for i, masses in successor_mass.items():

        support = origin_support[i]

        if support == 0:
            continue

        conditional_probs[i] = {
            j: mass / support
            for j, mass in masses.items()
        }

    return (
        global_next_prob,
        conditional_probs,
        origin_support,
    )


def compute_sequential_specificity(
    latest_basket,
    global_next_prob,
    conditional_probs,
    origin_support,
):
    """
    Measure how much knowing the latest basket changes the
    expected next-item distribution relative to the dataset's
    generic next-item distribution.

    For each origin item i:

        w_i = support_i / (support_i + lambda)

        P*(j | i)
            = w_i P(j | i)
            + (1 - w_i) Q(j)

    The latest-basket predictive distribution is:

        P_u(j)
            = mean_{i in latest basket} P*(j | i)

    sequential_specificity is:

        JS(P_u || Q)

    using log base 2, so JS lies in [0, 1].

    Interpretation
    --------------
    0:
        latest basket provides essentially no sequential
        information beyond generic population behavior.

    high:
        latest basket points toward a substantially different
        next-item distribution.
    """
    if (
        not latest_basket
        or not global_next_prob
    ):
        return np.nan

    basket_size = len(latest_basket)

    # P_u can be written as:
    #
    #     a * Q + sparse_extra
    #
    # which lets us compute JS without materializing a dense
    # probability vector over the entire item catalog.

    baseline_weight = 0.0
    extra_mass = Counter()

    for i in latest_basket:

        support = origin_support.get(i, 0)

        if support == 0 or i not in conditional_probs:
            weight = 0.0
        else:
            weight = (
                support
                / (support + SEQUENTIAL_SHRINKAGE)
            )

        baseline_weight += (
            (1.0 - weight) / basket_size
        )

        if weight > 0:

            scale = weight / basket_size

            for j, prob in conditional_probs[i].items():
                extra_mass[j] += scale * prob

    # If no item in the latest basket has transition evidence,
    # P_u == Q exactly.
    if not extra_mass:
        return 0.0

    # --------------------------------------------------------
    # JS contribution for items appearing in sparse_extra
    # --------------------------------------------------------

    kl_p = 0.0
    kl_q = 0.0

    q_mass_inside = 0.0

    for j, extra in extra_mass.items():

        q = global_next_prob.get(j, 0.0)

        # Any item present in a conditional successor
        # distribution should also occur in Q, but guard anyway.
        if q <= 0:
            continue

        q_mass_inside += q

        p = baseline_weight * q + extra

        m = 0.5 * (p + q)

        if p > 0:
            kl_p += p * np.log2(p / m)

        kl_q += q * np.log2(q / m)

    # --------------------------------------------------------
    # Aggregate contribution for all other catalog items.
    #
    # Outside extra_mass:
    #
    #       P_u(j) = baseline_weight * Q(j)
    #
    # so the KL ratios are constant and can be computed from
    # the total remaining Q mass.
    # --------------------------------------------------------

    q_mass_outside = max(
        0.0,
        1.0 - q_mass_inside,
    )

    a = baseline_weight

    if q_mass_outside > 0:

        if a > 0:
            kl_p += (
                a
                * q_mass_outside
                * np.log2(
                    (2.0 * a)
                    / (a + 1.0)
                )
            )

        kl_q += (
            q_mass_outside
            * np.log2(
                2.0 / (a + 1.0)
            )
        )

    js = 0.5 * (kl_p + kl_q)

    # Numerical protection
    return float(
        np.clip(js, 0.0, 1.0)
    )


# ============================================================
# Collaborative structure
# ============================================================

def build_idf_weighted_user_matrix(histories):
    """
    Build user-item vectors:

        x_ui = log(1 + count_ui) * IDF(i)

    where:

        IDF(i) =
            log((N + 1) / (df_i + 1)) + 1

    This discounts extremely common items so user similarity
    captures collaborative neighborhood structure rather than
    primarily shared popularity.
    """
    users = list(histories.keys())

    all_items = sorted({
        item
        for baskets in histories.values()
        for basket in baskets
        for item in basket
    })

    item_to_col = {
        item: idx
        for idx, item in enumerate(all_items)
    }

    rows = []
    cols = []
    values = []

    for row_idx, user in enumerate(users):

        counts = Counter(
            item
            for basket in histories[user]
            for item in basket
        )

        for item, count in counts.items():
            rows.append(row_idx)
            cols.append(item_to_col[item])
            values.append(count)

    raw_matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(users), len(all_items)),
        dtype=float,
    )

    if raw_matrix.shape[1] == 0:
        return users, raw_matrix

    # Number of users containing each item
    document_frequency = np.asarray(
        (raw_matrix > 0).sum(axis=0)
    ).ravel()

    n_users = len(users)

    idf = (
        np.log(
            (n_users + 1)
            / (document_frequency + 1)
        )
        + 1.0
    )

    weighted = raw_matrix.copy()

    # Log-frequency weighting
    weighted.data = np.log1p(weighted.data)

    # Column-wise IDF
    weighted = weighted.multiply(idf).tocsr()

    return users, weighted


def compute_neighbor_similarity(histories):

    print(
        f"{datetime.now()}:     "
        f"building IDF-weighted user-item matrix..."
    )

    users, matrix = build_idf_weighted_user_matrix(
        histories
    )

    if len(users) < 2:
        return {
            user: np.nan
            for user in users
        }

    if matrix.shape[1] == 0:
        return {
            user: np.nan
            for user in users
        }

    density = (
        matrix.nnz
        / (matrix.shape[0] * matrix.shape[1])
    )

    print(
        f"{datetime.now()}:     user-item matrix: "
        f"{matrix.shape[0]} users x "
        f"{matrix.shape[1]} items, "
        f"{matrix.nnz} nonzeros, "
        f"density={density:.6f}"
    )

    n_neighbors = min(
        TOP_NEIGHBORS + 1,
        len(users),
    )

    print(
        f"{datetime.now()}:     "
        f"fitting NearestNeighbors "
        f"(k={n_neighbors}, cosine/brute)..."
    )

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    model.fit(matrix)

    print(
        f"{datetime.now()}:     "
        f"querying nearest neighbors..."
    )

    distances, indices = model.kneighbors(matrix)

    similarities = 1.0 - distances

    result = {}

    for row_idx, user in enumerate(users):

        # First result should be the user itself.
        neighbor_sims = similarities[row_idx][1:]

        result[user] = (
            float(np.mean(neighbor_sims))
            if len(neighbor_sims) > 0
            else np.nan
        )

    print(
        f"{datetime.now()}:     "
        f"neighbor similarity done"
    )

    return result


# ============================================================
# Per-user features
# ============================================================

def compute_popularity_profile(
    baskets,
    popularity_percentiles,
):
    """
    Purchase-weighted average global popularity percentile.

    Repeated purchases count repeatedly because this feature
    describes the popularity profile of the user's actual
    consumption behavior.
    """
    counts = Counter(
        item
        for basket in baskets
        for item in basket
    )

    total = sum(counts.values())

    if total == 0:
        return np.nan

    weighted_sum = sum(
        count * popularity_percentiles[item]
        for item, count in counts.items()
    )

    return float(weighted_sum / total)


# ============================================================
# Main
# ============================================================

all_rows = []

output_path = (
    ROOT
    / "analysis"
    / "data"
    / "user_population_features.csv"
)

print(f"Writing to {output_path}")

print(
    f"{datetime.now()}: "
    f"Starting population features "
    f"for {len(DATASETS)} datasets"
)


for dataset_idx, dataset in enumerate(
    DATASETS,
    start=1,
):

    print()
    print("=" * 70)

    print(
        f"{datetime.now()}: "
        f"[{dataset_idx}/{len(DATASETS)}] "
        f"Processing {dataset}..."
    )

    # --------------------------------------------------------
    # Load histories
    # --------------------------------------------------------

    histories = load_histories(dataset)

    n_baskets = sum(
        len(baskets)
        for baskets in histories.values()
    )

    print(
        f"{datetime.now()}:   loaded "
        f"{len(histories)} users, "
        f"{n_baskets} observed baskets"
    )

    # --------------------------------------------------------
    # Popularity + relational statistics
    # --------------------------------------------------------

    print(
        f"{datetime.now()}:   "
        f"building basket statistics..."
    )

    (
        item_basket_counts,
        pair_counts,
        total_baskets,
    ) = build_basket_statistics(histories)

    print(
        f"{datetime.now()}:   "
        f"{len(item_basket_counts)} items, "
        f"{len(pair_counts)} observed pairs, "
        f"{total_baskets} non-empty baskets"
    )

    popularity_percentiles = (
        build_popularity_percentiles(
            item_basket_counts
        )
    )

    print(
        f"{datetime.now()}:   "
        f"building normalized pair associations..."
    )

    pair_associations = build_pair_associations(
        item_basket_counts=item_basket_counts,
        pair_counts=pair_counts,
        total_baskets=total_baskets,
    )

    # --------------------------------------------------------
    # Sequential statistics
    # --------------------------------------------------------

    print(
        f"{datetime.now()}:   "
        f"building transition statistics..."
    )

    (
        global_next_prob,
        conditional_probs,
        origin_support,
    ) = build_transition_statistics(histories)

    print(
        f"{datetime.now()}:   "
        f"sequential distributions available for "
        f"{len(conditional_probs)} origin items"
    )

    # --------------------------------------------------------
    # Collaborative structure
    # --------------------------------------------------------

    print(
        f"{datetime.now()}:   "
        f"computing collaborative neighborhoods..."
    )

    neighbor_similarity = (
        compute_neighbor_similarity(histories)
    )

    # --------------------------------------------------------
    # User-level regime features
    # --------------------------------------------------------

    print(
        f"{datetime.now()}:   "
        f"computing per-user features..."
    )

    for user, baskets in tqdm(
        histories.items(),
        total=len(histories),
        desc=f"{dataset} users",
    ):

        popularity_profile = (
            compute_popularity_profile(
                baskets=baskets,
                popularity_percentiles=
                    popularity_percentiles,
            )
        )

        relational_strength = (
            compute_relational_strength(
                baskets=baskets,
                pair_associations=
                    pair_associations,
            )
        )

        latest_basket = (
            baskets[-1]
            if baskets
            else set()
        )

        sequential_specificity = (
            compute_sequential_specificity(
                latest_basket=latest_basket,
                global_next_prob=
                    global_next_prob,
                conditional_probs=
                    conditional_probs,
                origin_support=
                    origin_support,
            )
        )

        all_rows.append({
            "dataset": dataset,
            "user": user,
            "popularity_profile":
                popularity_profile,
            "neighbor_similarity":
                neighbor_similarity[user],
            "sequential_specificity":
                sequential_specificity,
            "relational_strength":
                relational_strength,
        })

    print(
        f"{datetime.now()}:   "
        f"finished {dataset} "
        f"({len(all_rows)} total rows)"
    )


# ============================================================
# Save
# ============================================================

features = pd.DataFrame(all_rows)

features.to_csv(
    output_path,
    index=False,
)

print()
print("=" * 70)

print(
    f"{datetime.now()}: "
    f"Feature table shape: {features.shape}"
)

print("\nUsers by dataset:")
print(
    features.groupby("dataset").size()
)

print("\nMissing values:")
print(
    features.isna().sum()
)

print("\nDataset-level feature summary:")
print(
    features
    .groupby("dataset")[
        [
            "popularity_profile",
            "neighbor_similarity",
            "sequential_specificity",
            "relational_strength",
        ]
    ]
    .agg(["mean", "median", "std"])
    .round(4)
)

print(
    f"\n{datetime.now()}: "
    f"Saved to {output_path}"
)