#!/usr/bin/env python3
"""
Compute pairwise distances for SNP and STR genotypes from the same dataset.

- Ensures only samples that have BOTH SNP and STR data are used.
- STR: Goldstein δμ² distance
- SNP: Allele sharing distance (ASD), using (1/(2L)) * sum (x_i - x_j)^2

Input:
  - STR matrix TSV: loci x samples (entries = mean STR repeat lengths)
  - SNP matrix TSV: loci x samples (entries = 0/1/2 or similar)

Output:
  - <out_prefix>_str_goldstein_dist.tsv
  - <out_prefix>_snp_asd_dist.tsv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from utils.distance_utils import mantel_test

# ==============================
# Distance metrics
# ==============================

def goldstein_metric(u, v):
    """
    Goldstein δμ² distance between two individuals.
    Handles NaNs by ignoring loci where either is NaN.
    u, v: 1D arrays of STR allele lengths (mean repeat count per locus).
    """
    mask = ~np.isnan(u) & ~np.isnan(v)
    if not np.any(mask):
        return np.nan
    diff = u[mask] - v[mask]
    return np.mean(diff ** 2)


def goldstein_pdist(str_matrix_2d):
    """
    Compute Goldstein δμ² pairwise distances between individuals.

    Parameters
    ----------
    str_matrix_2d : numpy.ndarray
        Shape: (n_individuals, n_loci),
        entries = STR allele repeat lengths (mean if diploid).

    Returns
    -------
    dist_mat : numpy.ndarray
        Pairwise distance matrix of shape (n_individuals, n_individuals).
    """
    d = pdist(str_matrix_2d, metric=goldstein_metric)
    return squareform(d)


def asd_metric(u, v):
    """
    Allele Sharing Distance (ASD) between two individuals for SNP genotypes.

    For biallelic SNPs coded as 0/1/2 copies of an allele, ASD is:

        ASD(i, j) = (1 / (2L)) * Σ_l (x_il - x_jl)^2

    We implement this with NaN handling: ignore loci where either is NaN.
    """
    mask = ~np.isnan(u) & ~np.isnan(v)
    if not np.any(mask):
        return np.nan
    diff = u[mask] - v[mask]
    return np.mean(diff ** 2) / 2.0


def asd_pdist(snp_matrix_2d):
    """
    Compute allele sharing distance (ASD) pairwise distances between individuals.

    Parameters
    ----------
    snp_matrix_2d : numpy.ndarray
        Shape: (n_individuals, n_loci),
        entries = SNP genotypes (0/1/2 or similar).

    Returns
    -------
    dist_mat : numpy.ndarray
        Pairwise distance matrix of shape (n_individuals, n_individuals).
    """
    d = pdist(snp_matrix_2d, metric=asd_metric)
    return squareform(d)


# ==============================
# Main logic
# ==============================

def compute_pairwise_distances(str_matrix_path, snp_matrix_path, out_prefix):
    """
    Load STR and SNP matrices, align samples, compute pairwise distances, and save.
    """

    print(f"Loading STR matrix from: {str_matrix_path}")
    str_df = pd.read_csv(str_matrix_path, sep="\t", index_col=0)

    print(f"Loading SNP matrix from: {snp_matrix_path}")
    snp_df = pd.read_csv(snp_matrix_path, sep="\t", index_col=0)

    # Ensure samples overlap and align
    common_samples = str_df.columns.intersection(snp_df.columns)
    print(f"Found {len(common_samples)} samples with BOTH STR and SNP data.")

    if len(common_samples) == 0:
        raise ValueError("No overlapping samples between STR and SNP matrices!")

    str_df = str_df.loc[:, common_samples]
    snp_df = snp_df.loc[:, common_samples]

    # Convert to (n_individuals, n_loci) arrays
    # rows = individuals, cols = loci
    str_mat = str_df.T.values.astype(float)
    snp_mat = snp_df.T.values.astype(float)
    sample_ids = common_samples.to_list()

    # STR: Goldstein distance
    print("Computing Goldstein δμ² distances for STRs ...")
    str_dist = goldstein_pdist(str_mat)  # shape: (n_individuals, n_individuals)

    # SNP: ASD distance
    print("Computing allele sharing distances (ASD) for SNPs ...")
    snp_dist = asd_pdist(snp_mat)

    # Wrap into DataFrames with sample IDs as rows/cols
    str_dist_df = pd.DataFrame(str_dist, index=sample_ids, columns=sample_ids)
    snp_dist_df = pd.DataFrame(snp_dist, index=sample_ids, columns=sample_ids)

    # Save
    str_out = f"{out_prefix}_str_goldstein_dist.tsv"
    snp_out = f"{out_prefix}_snp_asd_dist.tsv"

    print(f"Writing STR distance matrix to: {str_out}")
    str_dist_df.to_csv(str_out, sep="\t")

    print(f"Writing SNP distance matrix to: {snp_out}")
    snp_dist_df.to_csv(snp_out, sep="\t")

    print("Done.")
    
    return str_dist_df, snp_dist_df


# ==============================
# CLI
# ==============================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Goldstein STR and ASD SNP pairwise distances "
                    "for a dataset (e.g. 1KGP, HGDP)."
    )
    p.add_argument(
        "--str-matrix",
        required=True,
        help="Path to STR genotype matrix TSV (loci x samples).",
    )
    p.add_argument(
        "--snp-matrix",
        required=True,
        help="Path to SNP genotype matrix TSV (loci x samples).",
    )
    p.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for distance matrix outputs, e.g. results/1kgp or results/hgdp.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    str_dist_df, snp_dist_df = compute_pairwise_distances(
        str_matrix_path=args.str_matrix,
        snp_matrix_path=args.snp_matrix,
        out_prefix=args.out_prefix,
    )
    
    r, p = mantel_test(str_dist_df, snp_dist_df, perms=10000, random_state=42)
    print("Mantel correlation:", r)
    print("Mantel p-value:", p)

if __name__ == "__main__":
    main()
