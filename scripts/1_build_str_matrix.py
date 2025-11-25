import os
import numpy as np
import pandas as pd


def normalize_chrom_label(chrom):
    """Return canonical autosome label as string, e.g. '1' for '1' or 'chr1'."""
    c = str(chrom).lower().replace("chr", "")
    return c


def build_str_matrix(out_dir, info_path, dataset, sep="\t", var_threshold=None):
    """
    Build a genome-wide STR matrix from per-chromosome matrices, with optional
    filtering on str_var.

    Parameters
    ----------
    out_dir : str
        Directory where per-chrom files and str_info live.
        Examples:
            "output"  (contains chr1_kg_matrix.tsv, kg_str_info.tsv, etc.)

    info_path : str
        Path to the global str_info file.
        Examples:
            "output/kg_str_info.tsv"
            "output/hg_str_info.tsv"

    dataset : str
        Dataset tag used in per-chrom filenames:
            - "kg"   -> chr1_kg_matrix.tsv, ...
            - "h3a"  -> chr1_h3a_matrix.tsv, ...
            - "hg"   -> chr1_hg_matrix.tsv, ...
            - "sg"   -> chr1_sg_matrix.tsv, ...

    sep : str, default "\t"
        Separator used in the TSV files.

    var_threshold : float or None, default None
        If not None, keep only STRs with str_var > var_threshold.
        Examples:
            var_threshold = 1.0
            var_threshold = 5.0

    Returns
    -------
    pandas.DataFrame
        Genome-wide STR matrix:
          - rows = STR loci (filtered by str_var if var_threshold given)
          - index = str_uid
          - columns = samples
    """

    # Load global str_info
    str_info = pd.read_csv(info_path, sep=sep)

    # Normalize chromosome labels
    str_info["chr_label"] = str_info["chrom"].apply(normalize_chrom_label)

    # Sort chromosomes numerically (1..22)
    def chr_key(c):
        c = str(c)
        return int(c) if c.isdigit() else np.inf

    chromosomes = sorted(str_info["chr_label"].unique(), key=chr_key)

    matrices = []

    for chr_label in chromosomes:
        fname = f"chr{chr_label}_{dataset}_matrix.tsv"
        fpath = os.path.join(out_dir, fname)

        if not os.path.exists(fpath):
            print(f"[WARN] Missing file for dataset '{dataset}', chr{chr_label}: {fpath}")
            continue

        print(f"Reading {fpath} ...")

        # Info for this chromosome
        info_chr = str_info[str_info["chr_label"] == chr_label].reset_index(drop=True)

        # Read per-chromosome matrix
        mat_chr = pd.read_csv(fpath, sep=sep, index_col=0)

        # Sanity check (optional)
        if mat_chr.shape[0] != info_chr.shape[0]:
            print(f"[WARN] Row count mismatch for chr{chr_label}: "
                  f"matrix {mat_chr.shape[0]} vs info {info_chr.shape[0]}")

        # Apply variability filter if requested
        if var_threshold is not None:
            mask = info_chr["str_var"] > var_threshold
            kept = mask.sum()
            total = len(mask)
            print(f"  chr{chr_label}: keeping {kept}/{total} STRs with str_var > {var_threshold}")
            info_chr = info_chr[mask].reset_index(drop=True)
            mat_chr = mat_chr.loc[mask.values, :]

        # Build unique STR IDs
        info_chr["str_uid"] = (
            info_chr["chrom"].astype(str)
            + ":" + info_chr["start"].astype(str)
            + "-" + info_chr["end"].astype(str)
        )

        # Assign row index
        mat_chr.index = info_chr["str_uid"].values

        matrices.append(mat_chr)

    if len(matrices) == 0:
        print(f"[WARN] No matrices built for dataset '{dataset}'. Returning empty DataFrame.")
        return pd.DataFrame()

    df_str = pd.concat(matrices, axis=0)

    return df_str

def main():
    kg_matrix_var1 = build_str_matrix(
    out_dir="output",
    info_path="output/kg_str_info.tsv",
    dataset="kg",
    sep="\t",
    var_threshold=1.0
)
    kg_matrix_var5 = build_str_matrix(
    out_dir="output",
    info_path="output/kg_str_info.tsv",
    dataset="kg",
    var_threshold=5.0
)  
    hg_matrix_var1 = build_str_matrix(
    out_dir="output",
    info_path="output/hg_str_info.tsv",
    dataset="hg",
    var_threshold=1.0
)
    #kg_matrix_var1.to_csv("output/kg_genomewide_var1_matrix.tsv", sep="\t")


if __name__ == "__main__":
    main()
