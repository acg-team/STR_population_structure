#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot stacked admixture barplots for true_Q, Q_pos, and Q_neg."
    )
    parser.add_argument("--q-pos", required=True, help="Q_pos TSV file.")
    parser.add_argument("--q-neg", required=True, help="Q_neg TSV file.")
    parser.add_argument(
        "--true-q",
        default=None,
        help="Optional true_Q TSV file to plot as the top panel.",
    )
    parser.add_argument(
        "--pos-mapping",
        default=None,
        help="Optional Q_pos to true_Q component mapping TSV from recover_q.py.",
    )
    parser.add_argument(
        "--neg-mapping",
        default=None,
        help="Optional Q_neg to true_Q component mapping TSV from recover_q.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output image path, for example admixture.png or admixture.pdf.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata TSV with individual and population columns.",
    )
    parser.add_argument(
        "--sort-by-pop",
        action="store_true",
        help="Sort individuals by metadata population, then by max ancestry instead of clustering.",
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Keep input order unless --sort-by-pop is used.",
    )
    parser.add_argument(
        "--cluster-metric",
        default="euclidean",
        help="Distance metric for hierarchical clustering.",
    )
    parser.add_argument(
        "--cluster-method",
        default="average",
        help="Linkage method for hierarchical clustering.",
    )
    parser.add_argument(
        "--show-pop-labels",
        action="store_true",
        help="Show true population labels and separators using metadata.",
    )
    parser.add_argument(
        "--pop-gap",
        type=float,
        default=3.0,
        help="Gap width, in bar units, between true population groups.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=14,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=6,
        help="Figure height in inches.",
    )
    return parser.parse_args()


def read_q(path):
    q = pd.read_csv(path, sep="\t")
    q.columns = [f"K{i + 1}" for i in range(q.shape[1])]
    return q


def read_true_q(path):
    q = pd.read_csv(path, sep="\t", header=None)
    q.columns = [f"K{i + 1}" for i in range(q.shape[1])]
    return q


def reorder_to_true_components(q, mapping_path):
    mapping = pd.read_csv(mapping_path, sep="\t")
    required = {"estimated_component", "true_component"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping file is missing columns: {sorted(missing)}")

    q_reordered = pd.DataFrame(index=q.index)
    for true_component in sorted(mapping["true_component"]):
        estimated_component = mapping.loc[
            mapping["true_component"] == true_component, "estimated_component"
        ].iloc[0]
        q_reordered[f"K{true_component}"] = q[f"K{estimated_component}"]

    return q_reordered


def get_cluster_order(q, metric="euclidean", method="average"):
    if q.shape[0] <= 1:
        return np.arange(q.shape[0])

    distances = pdist(q.to_numpy(), metric=metric)
    linkage_matrix = linkage(distances, method=method)
    return leaves_list(linkage_matrix)


def get_within_population_cluster_order(q, metadata, metric="euclidean", method="average"):
    ordered_indices = []
    sortable = metadata.copy()
    sortable["_index"] = np.arange(q.shape[0])

    for _, group in sortable.groupby("population", sort=True):
        group_indices = group["_index"].to_numpy()
        local_q = q.iloc[group_indices]
        local_order = get_cluster_order(local_q, metric=metric, method=method)
        ordered_indices.extend(group_indices[local_order])

    return np.array(ordered_indices)


def get_order(
    q,
    metadata=None,
    sort_by_pop=False,
    cluster=True,
    cluster_metric="euclidean",
    cluster_method="average",
):
    n = q.shape[0]
    order = np.arange(n)
    meta = None

    if metadata is not None:
        meta = pd.read_csv(metadata, sep="\t")
        if meta.shape[0] != n:
            raise ValueError(
                "Metadata and Q files have different numbers of individuals: "
                f"{meta.shape[0]} vs {n}"
            )

        if sort_by_pop:
            max_component = q.to_numpy().argmax(axis=1)
            sortable = meta.copy()
            sortable["_index"] = np.arange(n)
            sortable["_max_component"] = max_component
            sortable = sortable.sort_values(
                ["population", "_max_component", "_index"], kind="mergesort"
            )
            order = sortable["_index"].to_numpy()
            meta = meta.iloc[order].reset_index(drop=True)
        elif cluster and "population" in meta.columns:
            order = get_within_population_cluster_order(
                q,
                meta,
                metric=cluster_metric,
                method=cluster_method,
            )
            meta = meta.iloc[order].reset_index(drop=True)
        elif cluster:
            order = get_cluster_order(q, metric=cluster_metric, method=cluster_method)
            meta = meta.iloc[order].reset_index(drop=True)
        else:
            meta = meta.reset_index(drop=True)

    elif sort_by_pop:
        max_component = q.to_numpy().argmax(axis=1)
        order = np.lexsort((np.arange(n), max_component))
    elif cluster:
        order = get_cluster_order(q, metric=cluster_metric, method=cluster_method)

    return order, meta


def get_population_spans(metadata):
    if metadata is None or "population" not in metadata.columns:
        return []

    spans = []
    start = 0
    populations = metadata["population"].to_numpy()
    for i in range(1, len(populations) + 1):
        if i == len(populations) or populations[i] != populations[start]:
            spans.append((populations[start], start, i - 1))
            start = i
    return spans


def make_x_positions(n, population_spans=None, gap=0.0):
    x = np.arange(n, dtype=float)
    if not population_spans or gap <= 0:
        return x

    for group_index, (_, start, end) in enumerate(population_spans):
        x[start : end + 1] += group_index * gap
    return x


def plot_stacked_q(
    ax,
    q,
    colors,
    title,
    x,
    population_spans=None,
    show_pop_labels=False,
):
    bottom = np.zeros(q.shape[0])

    for i, col in enumerate(q.columns):
        values = q[col].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=1.0,
            color=colors[i],
            edgecolor="none",
            linewidth=0,
        )
        bottom += values

    ax.set_title(title)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ancestry proportion")
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if population_spans and show_pop_labels:
        for population, start, end in population_spans:
            if start > 0:
                boundary = (x[start - 1] + x[start]) / 2
                ax.axvline(boundary, color="black", linewidth=0.4, alpha=0.35)
            center = (x[start] + x[end]) / 2
            ax.text(
                center,
                -0.08,
                str(population),
                ha="center",
                va="top",
                fontsize=8,
                transform=ax.get_xaxis_transform(),
            )


def main():
    args = parse_args()

    q_pos = read_q(args.q_pos)
    q_neg = read_q(args.q_neg)
    q_true = read_true_q(args.true_q) if args.true_q is not None else None

    if args.pos_mapping is not None:
        q_pos = reorder_to_true_components(q_pos, args.pos_mapping)
    if args.neg_mapping is not None:
        q_neg = reorder_to_true_components(q_neg, args.neg_mapping)

    if q_pos.shape != q_neg.shape:
        raise ValueError(f"Q_pos and Q_neg shapes differ: {q_pos.shape} vs {q_neg.shape}")
    if q_true is not None and q_true.shape != q_pos.shape:
        raise ValueError(f"true_Q and inferred Q shapes differ: {q_true.shape} vs {q_pos.shape}")

    order, metadata = get_order(
        q_true if q_true is not None else q_pos,
        metadata=args.metadata,
        sort_by_pop=args.sort_by_pop,
        cluster=not args.no_cluster and not args.sort_by_pop,
        cluster_metric=args.cluster_metric,
        cluster_method=args.cluster_method,
    )
    if q_true is not None:
        q_true = q_true.iloc[order].reset_index(drop=True)
    q_pos = q_pos.iloc[order].reset_index(drop=True)
    q_neg = q_neg.iloc[order].reset_index(drop=True)
    population_spans = get_population_spans(metadata)
    x = make_x_positions(q_pos.shape[0], population_spans, gap=args.pop_gap)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(q_pos.shape[1])]

    n_panels = 3 if q_true is not None else 2
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(args.width, args.height),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    show_pop_labels = args.show_pop_labels or args.sort_by_pop or (
        args.metadata is not None and not args.no_cluster
    )
    panel_index = 0
    if q_true is not None:
        plot_stacked_q(
            axes[panel_index],
            q_true,
            colors,
            "true_Q",
            x,
            population_spans,
            show_pop_labels,
        )
        panel_index += 1
    plot_stacked_q(
        axes[panel_index],
        q_pos,
        colors,
        "Q_pos",
        x,
        population_spans,
        show_pop_labels,
    )
    panel_index += 1
    plot_stacked_q(
        axes[panel_index],
        q_neg,
        colors,
        "Q_neg",
        x,
        population_spans,
        show_pop_labels,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(q_pos.shape[1])
    ]
    labels = [f"true K{i + 1}" for i in range(q_pos.shape[1])]
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(q_pos.shape[1], 8),
        frameon=False,
    )

    out_path = Path(args.output)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved admixture plot to: {out_path}")


if __name__ == "__main__":
    main()
