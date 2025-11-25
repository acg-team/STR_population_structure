#!/usr/bin/env python3
"""
Evaluate K-means clustering performance (ARI) on SNP or STR genotypes from 1KGP.

Input:
  - A genotype matrix (SNP or STR) with:
        rows   = variants / loci
        columns= sample IDs (1KGP)
  - A sample metadata file with at least:
        'Superpopulation' and 'Population' columns

Output:
  - <out_prefix>_ari_long.tsv   : ARI for each repeat, each #PCs, each level
  - <out_prefix>_ari_summary.tsv: mean/std ARI per (#PCs, Level, Group)

Usage example:

  python 10_eval_kmeans_ari_1kgp.py \
      --matrix data/kg_str_genomewide_matrix.tsv \
      --sample-metadata data/1kgp_sample_metadata.tsv \
      --dataset-name 1kgp_STR \
      --num-pc 50 \
      --n-repeats 5 \
      --out-prefix results/1kgp_STR
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def evaluate_kmeans_ari_by_pcs(
    geno_matrix,
    sample_df,
    num_pc=50,
    n_repeats=5,
    dataset_name=None,
    random_state_base=0,
):
    """
    Evaluate K-means clustering performance (ARI) as a function of #PCs.

    Parameters
    ----------
    geno_matrix : pandas.DataFrame
        SNP or STR genotype matrix with shape (n_loci, n_samples).
        Rows = loci, columns = sample IDs.

    sample_df : pandas.DataFrame
        Sample metadata with index matching geno_matrix columns.
        Must contain at least:
            - 'Superpopulation'
            - 'Population'

    num_pc : int, default 50
        Maximum number of PCs to consider (1..num_pc).

    n_repeats : int, default 5
        Number of K-means runs per (Level, Group, Num_PCs) with different seeds.

    dataset_name : str or None, default None
        Optional label for the dataset (e.g. '1kgp_STR', '1kgp_SNP').

    random_state_base : int, default 0
        Base offset added to seeds (useful when comparing multiple datasets).

    Returns
    -------
    all_results_long : pandas.DataFrame
        Long-format table with columns:
            ['Dataset', 'Level', 'Group', 'Num_PCs', 'Repeat', 'ARI']

    all_results : pandas.DataFrame
        Aggregated table grouped by (Dataset, Level, Group, Num_PCs) with:
            'ARI_mean' and 'ARI_std'.
    """

    # Align samples between matrix and metadata
    common_samples = geno_matrix.columns.intersection(sample_df.index)
    geno_matrix = geno_matrix.loc[:, common_samples]
    sample = sample_df.loc[common_samples]

    # PCA on samples
    X = geno_matrix.T.values  # shape: (n_samples, n_loci)
    n_components = min(num_pc, X.shape[0], X.shape[1])

    pca = PCA(n_components=n_components, random_state=random_state_base)
    X_pca = pca.fit_transform(X)

    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    pca_data = pd.DataFrame(X_pca, index=common_samples, columns=pc_cols)

    # Precompute labels and indices
    num_pc = n_components
    seeds = range(random_state_base, random_state_base + n_repeats)

    superpop_labels = sample.loc[pca_data.index, "Superpopulation"]
    superpops = superpop_labels.unique().tolist()

    sp_indices = {
        sp: sample.index[sample["Superpopulation"] == sp] for sp in superpops
    }
    pop_labels_by_sp = {
        sp: sample.loc[sp_indices[sp], "Population"] for sp in superpops
    }
    n_clusters_by_sp = {sp: pop_labels_by_sp[sp].nunique() for sp in superpops}
    n_clusters_super = superpop_labels.nunique()

    rows = []

    # Loop over number of PCs
    for i in range(1, num_pc + 1):
        print(f"Evaluating with first {i} PCs")

        Xi = pca_data.iloc[:, :i]

        # ---- Superpopulation level (all samples) ----
        for r in seeds:
            km = KMeans(
                n_clusters=n_clusters_super,
                n_init=10,
                random_state=r,
                algorithm="elkan",
            ).fit(Xi)
            ari = adjusted_rand_score(superpop_labels, km.labels_)
            rows.append({
                "Dataset": dataset_name if dataset_name is not None else ".",
                "Level": "Superpopulation",
                "Group": "ALL",
                "Num_PCs": i,
                "Repeat": int(r - random_state_base),
                "ARI": float(ari),
            })

        # ---- Population level within each superpopulation ----
        for sp in superpops:
            Xi_sp = Xi.loc[sp_indices[sp]]
            y_sp = pop_labels_by_sp[sp]
            k_sp = n_clusters_by_sp[sp]

            for r in seeds:
                km = KMeans(
                    n_clusters=k_sp,
                    n_init=10,
                    random_state=r,
                    algorithm="elkan",
                ).fit(Xi_sp)
                ari = adjusted_rand_score(y_sp, km.labels_)
                rows.append({
                    "Dataset": dataset_name if dataset_name is not None else ".",
                    "Level": "Population",
                    "Group": sp,
                    "Num_PCs": i,
                    "Repeat": int(r - random_state_base),
                    "ARI": float(ari),
                })

    all_results_long = pd.DataFrame(rows)

    group_cols = ["Level", "Group", "Num_PCs"]
    if dataset_name is not None:
        group_cols = ["Dataset"] + group_cols

    all_results = (
        all_results_long
        .groupby(group_cols, as_index=False)
        .agg(ARI_mean=("ARI", "mean"), ARI_std=("ARI", "std"))
    )

    return all_results_long, all_results


# ================================================================
# CLI / main
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate K-means ARI on SNP/STR genotypes from 1KGP."
    )
    p.add_argument(
        "--matrix",
        required=True,
        help="Path to genome-wide genotype matrix (TSV), rows=loci, cols=samples.",
    )
    p.add_argument(
        "--sample-metadata",
        required=True,
        help="Path to sample metadata TSV with 'Superpopulation' and 'Population' columns.",
    )
    p.add_argument(
        "--dataset-name",
        default="1kgp",
        help="Name/label for this dataset (e.g., '1kgp_STR', '1kgp_SNP').",
    )
    p.add_argument(
        "--num-pc",
        type=int,
        default=50,
        help="Maximum number of PCs to evaluate (default: 50).",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of K-means repeats per setting (default: 5).",
    )
    p.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output files (e.g. results/1kgp_STR).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load matrix and metadata
    print(f"Loading genotype matrix from: {args.matrix}")
    geno_matrix = pd.read_csv(args.matrix, sep="\t", index_col=0)

    print(f"Loading sample metadata from: {args.sample_metadata}")
    sample_df = pd.read_csv(args.sample_metadata, sep="\t", index_col=0)

    # Evaluate K-means clustering
    all_long, all_summary = evaluate_kmeans_ari_by_pcs(
        geno_matrix=geno_matrix,
        sample_df=sample_df,
        num_pc=args.num_pc,
        n_repeats=args.n_repeats,
        dataset_name=args.dataset_name,
        random_state_base=0,
    )

    # Save results
    long_path = f"{args.out_prefix}_ari_long.tsv"
    summ_path = f"{args.out_prefix}_ari_summary.tsv"

    all_long.to_csv(long_path, sep="\t", index=False)
    all_summary.to_csv(summ_path, sep="\t", index=False)

    print(f"\nSaved per-repeat ARI results to: {long_path}")
    print(f"Saved summary ARI results to:   {summ_path}")


if __name__ == "__main__":
    main()
