from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"

DATASETS = ["dunnhumby", "instacart", "sam", "tafeng", "valuedshopper"]
METHODS = ["recency", "frequency", "tifuknn", "fpmc", "triple2vec", "dnntsp", "cbp", "diffrec"]
FEATURES = [
    "n_baskets", "avg_basket_size",
    "repeat_rate", "adjacent_basket_jaccard", "recent_basket_novelty",
    "popularity_profile", "neighbor_similarity", "transition_strength", "relational_strength",
]
DATASET_LABEL = {
    "dunnhumby": "Dunnhumby", "instacart": "Instacart", "sam": "SAM",
    "tafeng": "TaFeng", "valuedshopper": "ValuedShopper",
}
METHOD_LABEL = {
    "recency": "Recency", "frequency": "Frequency", "tifuknn": "TIFUKNN", "fpmc": "FPMC",
    "triple2vec": "Triple2Vec", "dnntsp": "DNNTSP", "cbp": "CBP", "diffrec": "DiffRec",
}
METHOD_COLOR = {
    "recency": "#7C3AED", "frequency": "#2B8CFF", "tifuknn": "#17B34A", "fpmc": "#4ADE80",
    "triple2vec": "#EAB308", "dnntsp": "#F97316", "cbp": "#DC2626", "diffrec": "#EC4899",
}
FEATURE_LABEL = {
    "n_baskets": "Depth",
    "avg_basket_size": "Basket\nsize",
    "repeat_rate": "Repeat\nrate",
    "adjacent_basket_jaccard": "Local\ncontinuity",
    "recent_basket_novelty": "Recent\nnovelty",
    "popularity_profile": "Popularity",
    "neighbor_similarity": "Neighbor\nsimilarity",
    "transition_strength": "Transition\nstrength",
    "relational_strength": "Relational\nstrength",
}
GROUPS = [
    ("History scale", 0, 2),
    ("Temporal behavior", 2, 5),
    ("Population & structural context", 5, 9),
]
VMAX = 0.08
GROUP_GAP = 0.46
TITLE_GAP = 1.72
CELL_W, CELL_H = 0.625, 0.142
CMAP = LinearSegmentedColormap.from_list("coef", [
    (0.00, "#1A56DB"), (0.30, "#5B93F5"), (0.44, "#D7E6FF"),
    (0.50, "#F7F8FA"),
    (0.56, "#FFD2C8"), (0.70, "#FF6A52"), (1.00, "#E10600"),
], N=256)


def fig3_behavior_competence():
    cells = pd.read_csv(DATA_DIR / "joint_regime_bootstrap_summary.csv")
    cells = cells.set_index(["dataset", "method", "feature"])

    x_centers, x = [], 0.5
    for j in range(len(FEATURES)):
        if j in (2, 5):
            x += GROUP_GAP
        x_centers.append(x)
        x += 1.0
    x_total = x_centers[-1] + 0.5

    rows, titles, seps, cursor = [], [], [], 0.0
    for i, dataset in enumerate(DATASETS):
        if i:
            seps.append(cursor + TITLE_GAP * 0.28)
        titles.append((dataset, cursor + TITLE_GAP * 0.64))
        cursor += TITLE_GAP
        for method in METHODS:
            rows.append((dataset, method, cursor + 0.5))
            cursor += 1.0
    y_total = cursor

    pad_x, pad_y = 0.013 / CELL_W, 0.011 / CELL_H
    ax_w, ax_h = x_total * CELL_W, y_total * CELL_H
    left, right, bottom, header = 0.98, 0.12, 0.94, 0.56
    fig_w, fig_h = left + ax_w + right, bottom + ax_h + header

    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes([left / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h])
    ax.set_xlim(0, x_total)
    ax.set_ylim(0, y_total)
    ax.set_axis_off()
    norm = Normalize(vmin=-VMAX, vmax=VMAX)
    mark_s = 18

    def draw_mark(artist, x, y, kind):
        if kind == "strong":
            artist.scatter(x, y, s=mark_s, c="#1a1a1a", edgecolors="white", linewidths=0.45, zorder=4, clip_on=False)
        elif kind == "moderate":
            artist.scatter(x, y, s=mark_s, facecolors="none", edgecolors="white", linewidths=1.0, zorder=3, clip_on=False)
            artist.scatter(x, y, s=mark_s, facecolors="none", edgecolors="#1a1a1a", linewidths=0.55, zorder=4, clip_on=False)
        elif kind == "mixed":
            artist.scatter(x, y, s=mark_s, marker="x", c="#4d4d4d", linewidths=0.7, zorder=4, clip_on=False)

    def fy(y_book):
        return (bottom + ((y_total - y_book) / y_total) * ax_h) / fig_h

    for dataset, method, y_book in rows:
        y = y_total - y_book
        for x, feature in zip(x_centers, FEATURES):
            row = cells.loc[(dataset, method, feature)]
            ax.add_patch(Rectangle(
                (x - 0.5 + pad_x, y - 0.5 + pad_y), 1 - 2 * pad_x, 1 - 2 * pad_y,
                facecolor=CMAP(norm(row.coef_mean)), edgecolor="none", zorder=2,
            ))
            draw_mark(ax, x, y, row.evidence)
        fig.text((left - 0.10) / fig_w, fy(y_book), METHOD_LABEL[method], ha="right", va="center", fontsize=6.4, color="#222222")

    for dataset, y_book in titles:
        fig.text((left - 0.10) / fig_w, fy(y_book), DATASET_LABEL[dataset], ha="right", va="center", fontsize=8.0, fontweight="bold", color="#111111")
    for y_book in seps:
        y = y_total - y_book
        ax.plot([0, x_total], [y, y], color="#8A8A8A", lw=0.9, zorder=3, solid_capstyle="butt")

    for boundary in (2, 5):
        x = x_centers[boundary] - 0.5 - GROUP_GAP / 2
        for dataset in DATASETS:
            block = [yb for d, _, yb in rows if d == dataset]
            ax.plot([x, x], [y_total - (max(block) + 0.5), y_total - (min(block) - 0.5)], color="#E3E3E3", lw=0.55, zorder=1)

    def header(x_data, dy, text, **kwargs):
        fig.text((left + (x_data / x_total) * ax_w) / fig_w, (bottom + ax_h + dy) / fig_h, text, ha="center", va="bottom", **kwargs)

    for x, feature in zip(x_centers, FEATURES):
        header(x, 0.025, FEATURE_LABEL[feature], fontsize=6.4, color="#222222", linespacing=1.05)
    for name, start, stop in GROUPS:
        x0, x1 = x_centers[start] - 0.5, x_centers[stop - 1] + 0.5
        y_line = (bottom + ax_h + 0.30) / fig_h
        fig.add_artist(plt.Line2D(
            [(left + (x0 / x_total) * ax_w) / fig_w, (left + (x1 / x_total) * ax_w) / fig_w],
            [y_line, y_line], transform=fig.transFigure, color="#4a4a4a", lw=0.7,
        ))
        header((x0 + x1) / 2, 0.325, name, fontsize=8.0, fontweight="bold", color="#111111")

    cbar_w = 2.85
    cax = fig.add_axes([left / fig_w, 0.48 / fig_h, cbar_w / fig_w, 0.082 / fig_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([-0.08, -0.04, 0, 0.04, 0.08])
    cb.set_ticklabels(["−0.08", "−0.04", "0", "+0.04", "+0.08"])
    cb.ax.tick_params(labelsize=6.3, length=2.2, width=0.4, pad=1.2)
    fig.text(left / fig_w, 0.62 / fig_h, "Mean coefficient across keysets", ha="left", va="bottom", fontsize=7.0)
    fig.text(
        left / fig_w, 0.06 / fig_h,
        "Coefficient = change in method-relative nDCG per +1 within-dataset SD, conditional on the other eight axes.",
        ha="left", va="bottom", fontsize=6.2, color="#333333",
    )

    key_w = fig_w - left - cbar_w - 0.30
    lax = fig.add_axes([(left + cbar_w + 0.22) / fig_w, 0.30 / fig_h, key_w / fig_w, 0.64 / fig_h])
    lax.set_xlim(0, key_w)
    lax.set_ylim(-0.16, 0.48)
    lax.set_axis_off()
    marks = [
        (0.36, "strong", "Bootstrap 95% CI excludes 0 in 3/3 keysets"),
        (0.20, "moderate", "Bootstrap 95% CI excludes 0 in 2/3 keysets"),
        (0.04, "mixed", "coefficient direction differs across keysets"),
        (-0.12, "none", "no symbol: same direction, ≤1/3 CIs exclude 0"),
    ]
    for y, kind, label in marks:
        draw_mark(lax, 0.08, y, kind)
        lax.text(0.22, y, label, ha="left", va="center", fontsize=6.15, color="#222222", clip_on=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "fig3_behavior_competence.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(path)


def fig2_marginal_response():
    columns = [
        ("dunnhumby", "n_baskets", "repeat_rate", "popularity_profile"),
        ("instacart", "avg_basket_size", "repeat_rate", "neighbor_similarity"),
        ("sam", "avg_basket_size", "adjacent_basket_jaccard", "transition_strength"),
        ("tafeng", "n_baskets", "recent_basket_novelty", "relational_strength"),
        ("valuedshopper", "avg_basket_size", "adjacent_basket_jaccard", "popularity_profile"),
    ]
    feature_label = {
        "n_baskets": "History depth",
        "avg_basket_size": "Average basket size",
        "repeat_rate": "Repeat rate",
        "adjacent_basket_jaccard": "Local continuity",
        "recent_basket_novelty": "Recent novelty",
        "popularity_profile": "Popularity profile",
        "neighbor_similarity": "Neighbor similarity",
        "transition_strength": "Transition strength",
        "relational_strength": "Relational strength",
    }
    group_title = ["History scale", "Temporal behavior", "Population & structural context"]

    features = sorted({feature for _, *feats in columns for feature in feats})
    performance = pd.read_csv(DATA_DIR / "performance.csv", dtype={"user": str})
    performance = performance[performance.dataset.isin(DATASETS)]
    regimes = pd.read_csv(DATA_DIR / "regime_features.csv", dtype={"user": str}, usecols=["dataset", "user"] + features)
    regimes = regimes[regimes.dataset.isin(DATASETS)]

    def response_curves(n_bins):
        binned = regimes.copy()
        for feature in features:
            binned[f"{feature}_bin"] = pd.NA
            for idx in binned.groupby("dataset").groups.values():
                x = binned.loc[idx, feature]
                valid = x.notna()
                if not valid.any():
                    continue
                pct = x.loc[valid].rank(method="average", pct=True)
                binned.loc[pct.index, f"{feature}_bin"] = np.ceil(pct.to_numpy() * n_bins).clip(1, n_bins).astype(int)
        analysis = performance.merge(binned, on=["dataset", "user"], how="left", validate="many_to_one")
        centered = analysis[METHODS].sub(analysis[METHODS].mean(axis=1), axis=0)
        centered.columns = [f"{method}_c" for method in METHODS]
        analysis = pd.concat([analysis, centered], axis=1)
        pieces = []
        for dataset, *feats in columns:
            block = analysis[analysis.dataset == dataset]
            for feature in feats:
                temp = block[block[feature].notna() & block[f"{feature}_bin"].notna()]
                for method in METHODS:
                    means = temp.groupby(["keyset", f"{feature}_bin"])[f"{method}_c"].mean().reset_index()
                    means["dataset"] = dataset
                    means["feature"] = feature
                    means["method"] = method
                    pieces.append(means.rename(columns={f"{feature}_bin": "bin", f"{method}_c": "y"}))
        curves = pd.concat(pieces, ignore_index=True)
        return curves.groupby(["dataset", "feature", "method", "bin"], as_index=False)["y"].mean()

    n_bins = 10
    curves = response_curves(n_bins)
    column_ylim = {}
    for dataset, *feats in columns:
        block = curves[(curves.dataset == dataset) & curves.feature.isin(feats)]
        lo, hi = float(block.y.min()), float(block.y.max())
        pad = max((hi - lo) * 0.12, 0.004)
        column_ylim[dataset] = (min(lo, 0) - pad, max(hi, 0) + pad)

    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "axes.unicode_minus": False})
    panel_w, panel_h = 1.28, 1.28
    col_gap, row_gap = 0.40, 0.26
    left, bottom, top, right = 0.96, 0.48, 0.36, 0.04
    fig_w = left + 5 * panel_w + 4 * col_gap + right
    fig_h = bottom + 3 * panel_h + 2 * row_gap + top
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    for c, (dataset, *feats) in enumerate(columns):
        y_lo, y_hi = column_ylim[dataset]
        yticks = [tick for tick in plt.MaxNLocator(4).tick_values(y_lo, y_hi) if y_lo <= tick <= y_hi]
        for r, feature in enumerate(feats):
            x = left + c * (panel_w + col_gap)
            y = bottom + (2 - r) * (panel_h + row_gap)
            ax = fig.add_axes([x / fig_w, y / fig_h, panel_w / fig_w, panel_h / fig_h])
            block = curves[(curves.dataset == dataset) & (curves.feature == feature)]
            ax.axhline(0, color="#6A6A6A", lw=0.7, zorder=2)
            for method in METHODS:
                series = block[block.method == method].sort_values("bin")
                ax.plot(
                    series["bin"], series["y"], color=METHOD_COLOR[method], lw=1.05,
                    marker="o", ms=3.0, markerfacecolor=METHOD_COLOR[method], markeredgewidth=0,
                    solid_capstyle="round", zorder=3,
                )
            ax.set_xlim(0.5, n_bins + 0.5)
            ax.set_xticks([1, 3, 5, 7, 10])
            ax.set_ylim(*column_ylim[dataset])
            ax.set_yticks(yticks)
            ax.set_title(feature_label[feature], fontsize=7.0, color="#222222", pad=2.2)
            ax.tick_params(labelsize=5.8, length=2.0, width=0.4, colors="#333333")
            for spine in ax.spines.values():
                spine.set_color("#444444")
                spine.set_linewidth(0.5)
            ax.grid(True, color="#E3E3E3", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
        cx = left + c * (panel_w + col_gap) + panel_w / 2
        fig.text(cx / fig_w, (bottom + 2 * (panel_h + row_gap) + panel_h + 0.24) / fig_h, DATASET_LABEL[dataset], ha="center", va="center", fontsize=8, fontweight="bold", color="#111111")

    for r, title in enumerate(group_title):
        row_mid = bottom + (2 - r) * (panel_h + row_gap) + panel_h / 2
        fig.text(0.36 / fig_w, row_mid / fig_h, title, rotation=90, ha="center", va="center", fontsize=6.4, fontweight="bold", color="#111111")
    stack_mid = bottom + (panel_h + row_gap) + panel_h / 2
    fig.text(0.12 / fig_w, stack_mid / fig_h, "Mean method-relative nDCG@5", rotation=90, ha="center", va="center", fontsize=7.5, color="#222222")
    x_mid = left + 2 * (panel_w + col_gap) + panel_w / 2
    fig.text(x_mid / fig_w, 0.28 / fig_h, "Quantile bin (low → high)", ha="center", va="center", fontsize=7.5, color="#222222")
    handles = [
        Line2D([0], [0], color=METHOD_COLOR[m], lw=1.3, marker="o", ms=3.4,
               markerfacecolor=METHOD_COLOR[m], markeredgewidth=0)
        for m in METHODS
    ]
    fig.legend(
        handles, [METHOD_LABEL[m] for m in METHODS],
        loc="lower center", ncol=8, frameon=False, fontsize=6.6,
        bbox_to_anchor=(x_mid / fig_w, 0.004), handletextpad=0.4, columnspacing=0.85,
        borderaxespad=0,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "fig2_marginal_response.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(path)


def fig1_aggregate_vs_competence():
    raw = pd.read_csv(DATA_DIR / "competence_winner_share.csv")
    raw = raw[raw.dataset.isin(DATASETS)]
    points = raw.groupby(["dataset", "method"], as_index=False).agg(
        mean_ndcg=("mean_ndcg", "mean"),
        winner_share=("winner_share_nonzero_users", "mean"),
    )

    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "axes.unicode_minus": False})
    panel, gap, left, bottom = 1.28, 0.42, 0.58, 0.58
    fig_w = left + 5 * panel + 4 * gap + 0.06
    fig_h = bottom + panel + 0.26
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    axes = [
        fig.add_axes([(left + i * (panel + gap)) / fig_w, bottom / fig_h, panel / fig_w, panel / fig_h])
        for i in range(5)
    ]

    for ax, dataset in zip(axes, DATASETS):
        block = points[points.dataset == dataset].set_index("method").loc[METHODS]
        ndcg_best = block.mean_ndcg.idxmax()
        share_best = block.winner_share.idxmax()
        for method in METHODS:
            if method == ndcg_best:
                marker, size = "*", 78
            elif method == share_best:
                marker, size = "^", 42
            else:
                marker, size = "o", 34
            ax.scatter(
                block.loc[method, "winner_share"], block.loc[method, "mean_ndcg"],
                s=size, c=METHOD_COLOR[method], marker=marker,
                edgecolors="none", linewidths=0, zorder=3,
            )
        lo, hi = block.mean_ndcg.min(), block.mean_ndcg.max()
        pad = max((hi - lo) * 0.14, 0.006)
        ax.set_xlim(-0.005, 0.215)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
        ax.set_xticklabels(["0", ".05", ".10", ".15", ".20"])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_title(DATASET_LABEL[dataset], fontsize=8, fontweight="bold", color="#111111", pad=3)
        ax.tick_params(labelsize=6.0, length=2.2, width=0.4, colors="#333333")
        for spine in ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.5)
        ax.grid(True, color="#E3E3E3", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean nDCG@5", fontsize=7.5, color="#222222")
    fig.text(
        (left + 2.5 * panel + 2 * gap) / fig_w, 0.33 / fig_h,
        "Fractional oracle-winner share among nonzero users",
        ha="center", va="center", fontsize=7.5, color="#222222",
    )
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.5,
               markerfacecolor=METHOD_COLOR[m], markeredgecolor="none")
        for m in METHODS
    ]
    shape_handles = [
        Line2D([0], [0], marker="*", linestyle="none", markersize=8, markerfacecolor="#333333", markeredgecolor="#333333"),
        Line2D([0], [0], marker="^", linestyle="none", markersize=5.5, markerfacecolor="#333333", markeredgecolor="#333333"),
    ]
    method_legend = fig.legend(
        method_handles, [METHOD_LABEL[m] for m in METHODS],
        loc="lower left", ncol=8, frameon=False, fontsize=6.4,
        bbox_to_anchor=(0, 0.012 / fig_h), handletextpad=0.28, columnspacing=0.75,
        borderaxespad=0,
    )
    shape_legend = fig.legend(
        shape_handles, ["Highest mean nDCG", "Highest winner share"],
        loc="lower left", ncol=2, frameon=False, fontsize=6.4,
        bbox_to_anchor=(0, 0.012 / fig_h), handletextpad=0.28, columnspacing=0.85,
        borderaxespad=0,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    method_box = method_legend.get_window_extent(renderer)
    shape_box = shape_legend.get_window_extent(renderer)
    gap = 0.42 * fig.dpi
    x0 = (fig.bbox.width - method_box.width - gap - shape_box.width) / 2
    legend_y = 0.02 / fig_h
    method_legend.set_bbox_to_anchor((x0 / fig.bbox.width, legend_y), transform=fig.transFigure)
    shape_legend.set_bbox_to_anchor(((x0 + method_box.width + gap) / fig.bbox.width, legend_y), transform=fig.transFigure)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "fig1_aggregate_vs_competence.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(path)


def fig4_target_classes():
    raw = pd.read_csv(DATA_DIR / "target_item_mechanisms.csv")
    raw = raw[(raw.analysis_type == "target_class") & raw.dataset.isin(DATASETS)]
    classes = ["last_basket_repeat", "older_repeat", "novel"]
    class_label = {"last_basket_repeat": "Immediate repeat", "older_repeat": "Older repeat", "novel": "Novel"}
    class_marker = {"last_basket_repeat": "o", "older_repeat": "^", "novel": "x"}
    hits = raw.groupby(["dataset", "method", "category"], as_index=False)["hit_rate_at5"].mean()
    shares = []
    for (dataset, keyset), block in raw[raw.method == METHODS[0]].groupby(["dataset", "keyset"]):
        total = block.n_items.sum()
        shares.append(block.assign(share=block.n_items / total)[["dataset", "category", "share"]])
    shares = pd.concat(shares).groupby(["dataset", "category"], as_index=False)["share"].mean()

    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "axes.unicode_minus": False})
    panel_w, panel_h, gap, left, bottom, top = 1.32, 1.32, 0.18, 0.78, 0.58, 0.52
    fig_w = left + 5 * panel_w + 4 * gap + 0.06
    fig_h = bottom + panel_h + top
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    centers = [(len(METHODS) - 1 - i) for i in range(len(METHODS))]

    for i, dataset in enumerate(DATASETS):
        ax = fig.add_axes([(left + i * (panel_w + gap)) / fig_w, bottom / fig_h, panel_w / fig_w, panel_h / fig_h])
        block = hits[hits.dataset == dataset]
        for m, method in enumerate(METHODS):
            for category in classes:
                value = float(block[(block.method == method) & (block.category == category)].hit_rate_at5.iloc[0])
                if value <= 1e-5:
                    continue
                ax.plot(
                    value, centers[m], linestyle="none", marker=class_marker[category],
                    color=METHOD_COLOR[method], markersize=5.2, markeredgewidth=1.05,
                    markerfacecolor=METHOD_COLOR[method], zorder=3,
                )
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(-0.55, centers[0] + 0.55)
        ax.set_yticks(centers)
        ax.set_yticklabels([METHOD_LABEL[m] for m in METHODS] if i == 0 else [])
        ax.xaxis.set_major_locator(plt.LogLocator(base=10))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: "1" if v >= 1 else rf"$10^{{{int(round(np.log10(v)))}}}$"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        mix = shares[shares.dataset == dataset].set_index("category").share
        subtitle = f"{100 * mix['last_basket_repeat']:.1f}% immediate · {100 * mix['older_repeat']:.1f}% older\n{100 * mix['novel']:.1f}% novel"
        ax.set_title(DATASET_LABEL[dataset], fontsize=8, fontweight="bold", color="#111111", pad=20)
        ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=5.4, color="#333333", annotation_clip=False)
        ax.tick_params(labelsize=6.0, length=2.0, width=0.4, colors="#333333")
        for spine in ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.5)
        ax.grid(True, color="#E3E3E3", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    fig.text((left + 2.5 * panel_w + 2 * gap) / fig_w, 0.36 / fig_h, "Hit rate@5 within target class (log scale)", ha="center", va="center", fontsize=7.5, color="#222222")
    handles = [
        Line2D([0], [0], linestyle="none", marker=class_marker[c], color="#333333", markersize=5.4, markeredgewidth=1.05, markerfacecolor="#333333")
        for c in classes
    ]
    fig.legend(handles, [class_label[c] for c in classes], loc="lower center", ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=((left + 2.5 * panel_w + 2 * gap) / fig_w, 0.004), handletextpad=0.35, columnspacing=1.3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "fig4_target_classes.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    fig1_aggregate_vs_competence()
    fig2_marginal_response()
    fig3_behavior_competence()
    fig4_target_classes()
