#!/usr/bin/env python3
"""
Preprocess STR calls from EnsembleTR or HipSTR VCFs.

Outputs:

EnsembleTR mode (--caller ensembletr)
  - One global:   kg_str_info.tsv
  - Per-chrom:    chr<chrom>_kg_matrix.tsv
                  chr<chrom>_h3a_matrix.tsv

HipSTR mode (--caller hipstr)
  - Tag derived from --out-prefix basename, e.g. HGDP -> 'hg', SGDP -> 'sg'
  - One global:   <tag>_str_info.tsv   e.g. hg_str_info.tsv
  - Per-chrom:    chr<chrom>_<tag>_matrix.tsv  e.g. chr1_hg_matrix.tsv

Usage:

  # EnsembleTR (HipSTR-derived EnsembleTR calls only)
  python 00_preprocess_strs.py \
      --caller ensembletr \
      --vcf-dir /path/to/ensembletr_vcfs \
      --sample-type /path/to/sample_type.csv \
      --out-prefix output/ensembletr   # only 'output/' will be used

  # HipSTR, HGDP
  python 00_preprocess_strs.py \
      --caller hipstr \
      --vcf-dir /path/to/hgdp_hipstr_vcfs \
      --out-prefix output/HGDP

  # HipSTR, SGDP
  python 00_preprocess_strs.py \
      --caller hipstr \
      --vcf-dir /path/to/sgdp_hipstr_vcfs \
      --out-prefix output/SGDP
"""

import os
import argparse
import numpy as np
import pandas as pd
from cyvcf2 import VCF


# ================================================================
# Helpers
# ================================================================

def mean_str_length(vals):
    """Convert NCOPY/GB values like ['4,5', '.', ...] into mean float array."""
    out = []
    for v in vals:
        if v in (None, "."):
            out.append(np.nan)
        else:
            parts = np.array(v.split(","), dtype=float)
            out.append(parts.mean())
    return np.array(out, dtype=np.float32)


def is_autosome(chrom):
    """
    Return True if chrom is an autosome (1–22), handling '1'/'chr1'.
    """
    c = str(chrom).lower().replace("chr", "")
    return c.isdigit() and 1 <= int(c) <= 22


def normalize_chrom_label(chrom):
    """Return canonical autosome label as string, e.g. '1' for '1' or 'chr1'."""
    c = str(chrom).lower().replace("chr", "")
    return c


# ================================================================
# EnsembleTR processing
# ================================================================

def process_ensembletr_vcf(vcf_path, list_1kg, list_h3a):
    """
    Process a single EnsembleTR VCF file (autosomes only).

    Returns:
      chrom_label (str or None),
      str_info (list of dict),
      kg_rows (list of arrays),
      h3a_rows (list of arrays)
    """

    if not os.path.exists(vcf_path):
        print(f"[WARN] Missing EnsembleTR file: {vcf_path}")
        return None, [], [], []

    try:
        vcf = VCF(vcf_path)
    except Exception as e:
        print(f"[WARN] Cannot open {vcf_path}: {e}")
        return None, [], [], []

    str_info = []
    kg_rows = []
    h3a_rows = []
    chrom_label = None

    for variant in vcf:

        # Keep only autosomes
        if not is_autosome(variant.CHROM):
            continue

        if chrom_label is None:
            chrom_label = normalize_chrom_label(variant.CHROM)

        # PERIOD filter
        period = variant.INFO.get("PERIOD")
        if period is None or period > 6:
            continue

        # METHODS filter: keep only HipSTR-derived calls (METHODS[2] == "1")
        methods = variant.INFO.get("METHODS", "")
        parts = methods.split("|")
        if len(parts) < 3 or parts[2] != "1":
            continue

        # NCOPY field
        ncopy_raw = variant.format("NCOPY")
        if ncopy_raw is None:
            continue

        ncopy = mean_str_length(ncopy_raw)

        # Split into cohorts
        kg_vals = ncopy[list_1kg]
        h3a_vals = ncopy[list_h3a]

        # Variance threshold (1KGP)
        var_kg = np.nanvar(kg_vals)
        if var_kg <= 0.2:
            continue

        kg_rows.append(kg_vals)
        h3a_rows.append(h3a_vals)

        str_info.append({
            "chrom": variant.CHROM,
            "start": variant.INFO.get("START"),
            "end": variant.INFO.get("END"),
            "period": period,
            "ru": variant.INFO.get("RU"),
            "methods": methods,
            "str_var": var_kg,
            "score_mean": np.nanmean(variant.format("SCORE"))
        })

    return chrom_label, str_info, kg_rows, h3a_rows


def run_ensembletr_pipeline(vcf_dir, sample_type_path, out_prefix):
    """
    Process all EnsembleTR VCFs (*.vcf.gz) in vcf_dir.

    Outputs:
      - kg_str_info.tsv (all chromosomes combined)
      - chr<chrom>_kg_matrix.tsv
      - chr<chrom>_h3a_matrix.tsv
    """

    print("\n=== Running EnsembleTR preprocessing (per chromosome matrices, global kg_str_info) ===")

    # Determine output directory
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Discover VCF files
    vcf_files = sorted(
        os.path.join(vcf_dir, f)
        for f in os.listdir(vcf_dir)
        if f.endswith(".vcf.gz")
    )

    if len(vcf_files) == 0:
        print(f"[ERROR] No *.vcf.gz files found in {vcf_dir}")
        return

    print(f"Found {len(vcf_files)} EnsembleTR VCF files:")
    for f in vcf_files:
        print("   >", f)

    # Load sample metadata
    meta = pd.read_csv(sample_type_path).dropna()
    sample_1kg = set(meta.query("Superpopulation != 'H3Africa'")["sample"])
    sample_h3a = set(meta.query("Superpopulation == 'H3Africa'")["sample"])

    # Use the first VCF to get full sample list / order
    first_vcf = VCF(vcf_files[0])
    all_samples = pd.Series(first_vcf.samples)

    list_1kg = all_samples[all_samples.isin(sample_1kg)].index.tolist()
    list_h3a = all_samples[all_samples.isin(sample_h3a)].index.tolist()

    print(f"Total samples: {len(all_samples)}")
    print(f" → 1KGP samples: {len(list_1kg)}")
    print(f" → H3Africa samples: {len(list_h3a)}")

    # Global str_info accumulator (locus-level, usable for both)
    all_str_info = []

    for vcf_path in vcf_files:
        print(f"\nProcessing {vcf_path} ...")

        chrom_label, info_chr, kg_chr, h3a_chr = process_ensembletr_vcf(
            vcf_path, list_1kg, list_h3a
        )

        # Append to global str_info
        all_str_info.extend(info_chr)

        # No passing variants in this file?
        if chrom_label is None or len(kg_chr) == 0:
            print("   [INFO] No passing autosomal STRs in this file.")
            continue

        # Build per-chromosome matrices
        kg_df = pd.DataFrame(kg_chr, columns=all_samples[list_1kg])
        h3a_df = pd.DataFrame(h3a_chr, columns=all_samples[list_h3a])

        # Per-chromosome output paths: chrX_kg_matrix.tsv, chrX_h3a_matrix.tsv
        kg_path = os.path.join(out_dir, f"chr{chrom_label}_kg_matrix.tsv")
        h3a_path = os.path.join(out_dir, f"chr{chrom_label}_h3a_matrix.tsv")

        kg_df.to_csv(kg_path, sep="\t")
        h3a_df.to_csv(h3a_path, sep="\t")

        print(f"   → {kg_path}")
        print(f"   → {h3a_path}")

    # Write global str_info once, named kg_str_info.tsv
    str_info_df = pd.DataFrame(all_str_info)
    info_path = os.path.join(out_dir, "kg_str_info.tsv")
    str_info_df.to_csv(info_path, sep="\t", index=False)
    print(f"\nGlobal STR info written to: {info_path}")


# ================================================================
# HipSTR processing
# ================================================================

def process_hipstr_vcf(vcf_path):
    """
    Process a single HipSTR VCF file (autosomes only).

    Returns:
      chrom_label (str or None),
      str_info (list of dict),
      rows (list of arrays),
      samples (array or None)
    """

    if not os.path.exists(vcf_path):
        print(f"[WARN] Missing HipSTR file: {vcf_path}")
        return None, [], [], None

    try:
        vcf = VCF(vcf_path)
    except Exception as e:
        print(f"[WARN] Cannot open {vcf_path}: {e}")
        return None, [], [], None

    samples = np.array(vcf.samples)

    info_chr = []
    rows_chr = []
    chrom_label = None

    for variant in vcf:

        # Keep only autosomes
        if not is_autosome(variant.CHROM):
            continue

        if chrom_label is None:
            chrom_label = normalize_chrom_label(variant.CHROM)

        period = variant.INFO.get("PERIOD")
        if period is None or period > 6:
            continue

        try:
            ref_len = round((variant.end - variant.start) / period)
        except Exception:
            continue

        gb_raw = variant.format("GB")
        if gb_raw is None:
            continue

        try:
            ncopy = mean_str_length(gb_raw) + ref_len
        except Exception:
            continue

        str_var = np.nanvar(ncopy)
        if str_var <= 0.1:
            continue

        rows_chr.append(ncopy)
        info_chr.append({
            "chrom": variant.CHROM,
            "start": variant.start,
            "end": variant.end,
            "period": period,
            "ref": variant.REF,
            "ref_len": ref_len,
            "str_var": str_var
        })

    return chrom_label, info_chr, rows_chr, samples


def tag_from_outprefix(out_prefix):
    """
    Derive a short tag from out-prefix basename.
    Examples:
      out-prefix 'output/HGDP' -> 'hg'
      out-prefix 'output/SGDP' -> 'sg'
      otherwise: lowercased basename (e.g. 'mytag' -> 'mytag')
    """
    base = os.path.basename(out_prefix)
    code = base.lower()
    if code == "hgdp":
        return "hg"
    if code == "sgdp":
        return "sg"
    return code


def run_hipstr_pipeline(vcf_dir, out_prefix):
    """
    Process all HipSTR VCFs (*.vcf.gz) in vcf_dir.

    Outputs:
      - <tag>_str_info.tsv (all chromosomes combined)
      - chr<chrom>_<tag>_matrix.tsv (per-chromosome matrices)

    Where <tag> is derived from basename(out_prefix), e.g.:
      out-prefix 'output/HGDP' -> 'hg'
      out-prefix 'output/SGDP' -> 'sg'
    """

    print("\n=== Running HipSTR preprocessing (per chromosome matrices, global <tag>_str_info) ===")

    # Determine output directory and tag
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    tag = tag_from_outprefix(out_prefix)

    vcf_files = sorted(
        os.path.join(vcf_dir, f)
        for f in os.listdir(vcf_dir)
        if f.endswith(".vcf.gz")
    )

    if len(vcf_files) == 0:
        print(f"[ERROR] No *.vcf.gz files found in {vcf_dir}")
        return

    print(f"Found {len(vcf_files)} HipSTR VCF files:")
    for f in vcf_files:
        print("   >", f)

    sample_names = None
    all_str_info = []

    for vcf_path in vcf_files:
        print(f"\nProcessing {vcf_path} ...")

        chrom_label, info_chr, rows_chr, samples_chr = process_hipstr_vcf(vcf_path)

        # Append to global str_info
        all_str_info.extend(info_chr)

        if chrom_label is None or len(rows_chr) == 0:
            print("   [INFO] No passing autosomal STRs in this file.")
            continue

        if sample_names is None and samples_chr is not None:
            sample_names = samples_chr

        str_matrix_df = pd.DataFrame(rows_chr, columns=sample_names)

        matrix_path = os.path.join(out_dir, f"chr{chrom_label}_{tag}_matrix.tsv")
        str_matrix_df.to_csv(matrix_path, sep="\t")

        print(f"   → {matrix_path}")

    # Write global str_info once, named <tag>_str_info.tsv
    str_info_df = pd.DataFrame(all_str_info)
    info_path = os.path.join(out_dir, f"{tag}_str_info.tsv")
    str_info_df.to_csv(info_path, sep="\t", index=False)
    print(f"\nGlobal STR info written to: {info_path}")


# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess EnsembleTR or HipSTR STR VCFs")
    p.add_argument(
        "--caller",
        required=True,
        choices=["ensembletr", "hipstr"],
        help="Which STR caller to preprocess.",
    )
    p.add_argument(
        "--vcf-dir",
        required=True,
        help="Directory containing *.vcf.gz files.",
    )
    p.add_argument(
        "--sample-type",
        help="Sample type CSV file (required for EnsembleTR).",
    )
    p.add_argument(
        "--out-prefix",
        required=True,
        help="For ensembletr: used only to determine output directory.\n"
             "For hipstr: basename(out-prefix) used as dataset tag (e.g. HGDP, SGDP).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.caller == "ensembletr":
        if args.sample_type is None:
            raise ValueError("--sample-type is required for EnsembleTR mode.")
        run_ensembletr_pipeline(args.vcf_dir, args.sample_type, args.out_prefix)

    elif args.caller == "hipstr":
        run_hipstr_pipeline(args.vcf_dir, args.out_prefix)


if __name__ == "__main__":
    main()
