#!/usr/bin/env python3
"""
Overlap STR loci between two datasets and filter loci with strong batch effects.

Works for, e.g.:
  - 1KGP + H3Africa
  - HGDP + SGDP
  - 1KGP + (HGDP+SGDP)

Inputs:
  - Two STR info tables (info1, info2) with at least:
        chr, start, end, str_id, str_var
  - Two genotype matrices (matrix1, matrix2):
        loci x samples, rows = loci, columns = sample IDs
        row index must correspond to STR IDs (str_uid)
  - Two sample metadata tables (meta1, meta2) with:
        sample ID column
        population/superpopulation column for grouping

Pipeline:
  1. Filter each STR set by str_var threshold.
  2. Use PyRanges to find overlapping loci (by genomic position).
  3. Build overlapping genotype matrices (same loci order in both datasets).
  4. For each population in pop_list:
       - compute mean length per locus in dataset1 and dataset2
       - compute difference
       - keep loci where |mean1 - mean2| < diff_threshold
     Combine across populations by intersection.
  5. Save cleaned STR info and matrices.
"""

import argparse
import numpy as np
import pandas as pd
import pyranges as pr


# ================================================================
# Overlap finder
# ================================================================

def load_str_info(path, sep="\t"):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        return pd.read_csv(path, sep=sep)


def find_overlapping_strs(
    df1,
    df2,
    var_thresh1=0.0,
    var_thresh2=0.0,
    chr_col="chr",
    start_col="start",
    end_col="end",
    id_col="str_id",
    var_col="str_var",
):
    """
    Find overlapping STR loci between two datasets using PyRanges.
    Returns:
      comb_df, df1_overlap, df2_overlap
    """

    # Filter by variance thresholds
    df1_f = df1.query(f"{var_col} > @var_thresh1").copy()
    df2_f = df2.query(f"{var_col} > @var_thresh2").copy()

    print(f"Dataset1: {len(df1)} total STRs, {len(df1_f)} with {var_col} > {var_thresh1}")
    print(f"Dataset2: {len(df2)} total STRs, {len(df2_f)} with {var_col} > {var_thresh2}")

    # Assign unique ID per STR if not present
    df1_f["str_uid"] = df1_f[chr_col].astype(str) + "_" + df1_f[id_col].astype(str)
    df2_f["str_uid"] = df2_f[chr_col].astype(str) + "_" + df2_f[id_col].astype(str)

    A = pr.PyRanges(pd.DataFrame({
        "Chromosome": df1_f[chr_col],
        "Start": df1_f[start_col],
        "End": df1_f[end_col],
        "id_1": df1_f["str_uid"],
    }))

    B = pr.PyRanges(pd.DataFrame({
        "Chromosome": df2_f[chr_col],
        "Start": df2_f[start_col],
        "End": df2_f[end_col],
        "id_2": df2_f["str_uid"],
    }))

    print("Finding overlaps with PyRanges.join() ...")
    comb = A.join(B)   # overlapping intervals
    comb_df = comb.df.reset_index(drop=True)

    if comb_df.empty:
        print("[WARN] No overlapping STR loci found.")
        return comb_df, None, None

    # Map back to original info tables
    df1_idxed = df1_f.set_index("str_uid")
    df2_idxed = df2_f.set_index("str_uid")

    df1_overlap = df1_idxed.loc[comb_df["id_1"]].reset_index()
    df2_overlap = df2_idxed.loc[comb_df["id_2"]].reset_index()

    print(f"Found {df1_overlap.shape[0]} overlapping STR pairs.")
    return comb_df, df1_overlap, df2_overlap


# ================================================================
# Batch-effect filter
# ================================================================

def filter_batch_effect_strs(
    mat1,
    mat2,
    meta1,
    meta2,
    pop_list,
    meta1_pop_col,
    meta1_sample_col,
    meta2_pop_col,
    meta2_sample_col,
    diff_threshold=1.0,
):
    """
    Filter overlapping STR loci by consistency across datasets:
    keep loci where |mean(len_dataset1) - mean(len_dataset2)| < diff_threshold
    for ALL populations in pop_list.

    mat1, mat2: loci x samples DataFrames (same loci order).
    meta1, meta2: metadata DataFrames for dataset1 and dataset2.

    Returns:
      keep_idx: numpy array of loci indices to keep (relative to mat1/mat2).
    """

    n_loci = mat1.shape[0]
    keep_indices = np.arange(n_loci)

    for p in pop_list:
        print(f"  Checking batch effect for population: {p}")

        # Samples of this population in dataset 1
        samples1 = meta1.loc[meta1[meta1_pop_col] == p, meta1_sample_col].values
        # Samples of this population in dataset 2
        samples2 = meta2.loc[meta2[meta2_pop_col] == p, meta2_sample_col].values

        # Intersect with columns present in matrices
        samples1 = [s for s in samples1 if s in mat1.columns]
        samples2 = [s for s in samples2 if s in mat2.columns]

        if len(samples1) == 0 or len(samples2) == 0:
            print(f"    [WARN] No samples for population {p} in one or both datasets, skipping this pop.")
            continue

        sub1 = mat1.loc[:, samples1]
        sub2 = mat2.loc[:, samples2]

        pop_mean = pd.DataFrame({
            "d1": sub1.mean(axis=1).values,
            "d2": sub2.mean(axis=1).values,
        })
        pop_mean["diff"] = pop_mean["d1"] - pop_mean["d2"]

        ok = pop_mean.index[pop_mean["diff"].abs() < diff_threshold].values

        keep_indices = np.intersect1d(keep_indices, ok)

    return keep_indices


# ================================================================
# CLI and main
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Overlap STR loci between two datasets and filter batch-effect loci."
    )
    # STR info
    p.add_argument("--info1", required=True,
                   help="STR info file for dataset1 (TSV/CSV).")
    p.add_argument("--info2", required=True,
                   help="STR info file for dataset2 (TSV/CSV).")
    p.add_argument("--var-thresh1", type=float, default=0.0,
                   help="Variance threshold for dataset1 (keep str_var > this).")
    p.add_argument("--var-thresh2", type=float, default=0.0,
                   help="Variance threshold for dataset2 (keep str_var > this).")
    p.add_argument("--sep", type=str, default="\t",
                   help="Separator for TSV files (ignored for CSV).")

    # Genotype matrices (loci x samples)
    p.add_argument("--matrix1", required=True,
                   help="Genotype matrix for dataset1 (TSV; loci x samples).")
    p.add_argument("--matrix2", required=True,
                   help="Genotype matrix for dataset2 (TSV; loci x samples).")

    # Metadata
    p.add_argument("--meta1", required=True,
                   help="Sample metadata for dataset1 (TSV).")
    p.add_argument("--meta2", required=True,
                   help="Sample metadata for dataset2 (TSV).")
    p.add_argument("--meta1-sample-col", required=True,
                   help="Column in meta1 with sample IDs.")
    p.add_argument("--meta1-pop-col", required=True,
                   help="Column in meta1 with population/superpopulation labels.")
    p.add_argument("--meta2-sample-col", required=True,
                   help="Column in meta2 with sample IDs.")
    p.add_argument("--meta2-pop-col", required=True,
                   help="Column in meta2 with population/superpopulation labels.")

    # Populations & threshold
    p.add_argument("--pop-list", type=str, required=True,
                   help="Comma-separated list of populations to use for batch-effect filter (e.g. 'AFR,EAS,EUR').")
    p.add_argument("--diff-threshold", type=float, default=1.0,
                   help="Absolute mean difference threshold for batch filter (default: 1.0).")

    # Output
    p.add_argument("--out-prefix", required=True,
                   help="Output prefix (e.g. results/kgp_h3a_overlap).")

    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------
    # Load STR info tables
    # ------------------------
    print(f"Loading STR info for dataset1 from: {args.info1}")
    info1 = load_str_info(args.info1, sep=args.sep)
    print(f"Loading STR info for dataset2 from: {args.info2}")
    info2 = load_str_info(args.info2, sep=args.sep)

    # ------------------------
    # Find overlapping STRs
    # ------------------------
    comb_df, info1_ov, info2_ov = find_overlapping_strs(
        info1,
        info2,
        var_thresh1=args.var_thresh1,
        var_thresh2=args.var_thresh2,
        chr_col="chr",
        start_col="start",
        end_col="end",
        id_col="str_id",
        var_col="str_var",
    )

    if info1_ov is None or info2_ov is None:
        print("No overlaps found; exiting.")
        return

    # ------------------------
    # Load genotype matrices
    # ------------------------
    print(f"Loading genotype matrix1 from: {args.matrix1}")
    mat1 = pd.read_csv(args.matrix1, sep="\t", index_col=0)
    print(f"Loading genotype matrix2 from: {args.matrix2}")
    mat2 = pd.read_csv(args.matrix2, sep="\t", index_col=0)

    # Matrices are loci x samples; row indices must match str_uid
    # We constructed str_uid in info*_ov as "str_uid"
    # Ensure overlapping order follows comb_df's id_1, id_2
    loci1 = info1_ov["str_uid"].values
    loci2 = info2_ov["str_uid"].values

    # Subset matrices to overlapping loci
    missing1 = [l for l in loci1 if l not in mat1.index]
    missing2 = [l for l in loci2 if l not in mat2.index]
    if missing1:
        print(f"[WARN] {len(missing1)} overlapping loci not found in matrix1; they will be dropped.")
    if missing2:
        print(f"[WARN] {len(missing2)} overlapping loci not found in matrix2; they will be dropped.")

    loci1_present = [l for l in loci1 if l in mat1.index]
    loci2_present = [l for l in loci2 if l in mat2.index]

    # Align by position in comb_df: assume one-to-one mapping and that
    # loci1_present and loci2_present are still aligned.
    mat1_ov = mat1.loc[loci1_present, :]
    mat2_ov = mat2.loc[loci2_present, :]

    # ------------------------
    # Load metadata
    # ------------------------
    print(f"Loading metadata1 from: {args.meta1}")
    meta1 = pd.read_csv(args.meta1, sep="\t")
    print(f"Loading metadata2 from: {args.meta2}")
    meta2 = pd.read_csv(args.meta2, sep="\t")

    # ------------------------
    # Batch-effect filtering
    # ------------------------
    pop_list = [p.strip() for p in args.pop_list.split(",") if p.strip()]
    print(f"Populations used for batch-effect filtering: {pop_list}")
    print(f"Using diff_threshold = {args.diff_threshold}")

    keep_idx = filter_batch_effect_strs(
        mat1=mat1_ov,
        mat2=mat2_ov,
        meta1=meta1,
        meta2=meta2,
        pop_list=pop_list,
        meta1_pop_col=args.meta1_pop_col,
        meta1_sample_col=args.meta1_sample_col,
        meta2_pop_col=args.meta2_pop_col,
        meta2_sample_col=args.meta2_sample_col,
        diff_threshold=args.diff_threshold,
    )

    print(f"Overlapping loci before batch filter: {mat1_ov.shape[0]}")
    print(f"Loci kept after batch filter: {len(keep_idx)}")

    mat1_clean = mat1_ov.iloc[keep_idx, :]
    mat2_clean = mat2_ov.iloc[keep_idx, :]

    # Need to subset info tables accordingly
    info1_clean = info1_ov.set_index("str_uid").loc[mat1_clean.index].reset_index()
    info2_clean = info2_ov.set_index("str_uid").loc[mat2_clean.index].reset_index()

    # ------------------------
    # Save outputs
    # ------------------------
    out_info1 = f"{args.out_prefix}_dataset1_str_info_clean.tsv"
    out_info2 = f"{args.out_prefix}_dataset2_str_info_clean.tsv"
    out_mat1 = f"{args.out_prefix}_dataset1_matrix_clean.tsv"
    out_mat2 = f"{args.out_prefix}_dataset2_matrix_clean.tsv"
    out_summary = f"{args.out_prefix}_summary.tsv"

    print(f"Saving cleaned STR info for dataset1 to: {out_info1}")
    info1_clean.to_csv(out_info1, sep="\t", index=False)

    print(f"Saving cleaned STR info for dataset2 to: {out_info2}")
    info2_clean.to_csv(out_info2, sep="\t", index=False)

    print(f"Saving cleaned genotype matrix1 to: {out_mat1}")
    mat1_clean.to_csv(out_mat1, sep="\t")

    print(f"Saving cleaned genotype matrix2 to: {out_mat2}")
    mat2_clean.to_csv(out_mat2, sep="\t")

    summary = pd.DataFrame([{
        "n_overlap_before": mat1_ov.shape[0],
        "n_overlap_after": len(keep_idx),
        "var_thresh1": args.var_thresh1,
        "var_thresh2": args.var_thresh2,
        "diff_threshold": args.diff_threshold,
        "pop_list": ",".join(pop_list),
    }])
    print(f"Saving summary to: {out_summary}")
    summary.to_csv(out_summary, sep="\t", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
