#!/usr/bin/env python3
"""
Unified preprocessing pipeline for EnsembleTR or HipSTR VCFs (chr1–22).

Usage:

  # EnsembleTR
  python 00_preprocess_strs.py \
      --caller ensembletr \
      --vcf-dir /path/to/ensembletr \
      --sample-type /path/to/sample_type.csv \
      --out-prefix output/ensembletr

  # HipSTR
  python 00_preprocess_strs.py \
      --caller hipstr \
      --vcf-dir /path/to/hipstr \
      --out-prefix output/hipstr
"""

import os
import argparse
import numpy as np
import pandas as pd
from cyvcf2 import VCF


# ================================================================
# Helper
# ================================================================

def mean_str_length(vals):
    """Convert NCOPY or GB entries into mean repeat lengths."""
    out = []
    for v in vals:
        if v in (None, "."):
            out.append(np.nan)
        else:
            parts = np.array(v.split(","), dtype=float)
            out.append(parts.mean())
    return np.array(out, dtype=np.float32)


# ================================================================
# EnsembleTR processing
# ================================================================

def process_ensembletr_chr(vcf_path, list_1kg, list_h3a, all_samples):
    if not os.path.exists(vcf_path):
        print(f"[WARN] Missing EnsembleTR file: {vcf_path}")
        return [], [], []

    try:
        vcf = VCF(vcf_path)
    except Exception:
        print(f"[WARN] Cannot open file: {vcf_path}")
        return [], [], []

    str_info = []
    kg_rows = []
    h3a_rows = []

    for variant in vcf:

        # PERIOD filter
        period = variant.INFO.get("PERIOD")
        if period is None or period > 6:
            continue

        # METHODS filter
        methods = variant.INFO.get("METHODS", "")
        parts = methods.split("|")
        if len(parts) < 4 or (parts[2] != "1" and parts[3] != "1"):
            continue

        # NCOPY field
        ncopy_raw = variant.format("NCOPY")
        if ncopy_raw is None:
            continue

        ncopy = mean_str_length(ncopy_raw)
        kg_vals = ncopy[list_1kg]
        h3a_vals = ncopy[list_h3a]

        # variance threshold
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

    return str_info, kg_rows, h3a_rows


def run_ensembletr_pipeline(vcf_dir, sample_type_path, out_prefix):
    print("\n=== Running EnsembleTR preprocessing ===")
    
    meta = pd.read_csv(sample_type_path).dropna()
    sample_1kg = set(meta.query("Superpopulation != 'H3Africa'")["sample"])
    sample_h3a = set(meta.query("Superpopulation == 'H3Africa'")["sample"])

    # Use chr1 to read sample order
    vcf_chr1 = VCF(os.path.join(vcf_dir, "ensemble_chr1_filtered.vcf.gz"))
    all_samples = pd.Series(vcf_chr1.samples)

    list_1kg = all_samples[all_samples.isin(sample_1kg)].index.tolist()
    list_h3a = all_samples[all_samples.isin(sample_h3a)].index.tolist()

    full_info = []
    full_kg = []
    full_h3a = []

    for chrom in range(1, 23):
        path = os.path.join(vcf_dir, f"ensemble_chr{chrom}_filtered.vcf.gz")
        print(f" → {path}")
        info_chr, kg_chr, h3a_chr = process_ensembletr_chr(path, list_1kg, list_h3a, all_samples)
        full_info.extend(info_chr)
        full_kg.extend(kg_chr)
        full_h3a.extend(h3a_chr)

    # Convert to DataFrame
    df_info = pd.DataFrame(full_info)
    df_kg   = pd.DataFrame(full_kg, columns=all_samples[list_1kg])
    df_h3a  = pd.DataFrame(full_h3a, columns=all_samples[list_h3a])

    # Output
    df_info.to_csv(f"{out_prefix}_str_info.tsv", sep="\t", index=False)
    df_kg.to_csv(f"{out_prefix}_1kg_matrix.tsv", sep="\t")
    df_h3a.to_csv(f"{out_prefix}_h3a_matrix.tsv", sep="\t")

    print("\nEnsembleTR preprocessing complete.\n")


# ================================================================
# HipSTR processing
# ================================================================

def process_hipstr_chr(vcf_path):
    if not os.path.exists(vcf_path):
        print(f"[WARN] Missing HipSTR file: {vcf_path}")
        return [], [], None

    vcf = VCF(vcf_path)
    samples = np.array(vcf.samples)

    info_chr = []
    rows_chr = []

    for variant in vcf:

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

    return info_chr, rows_chr, samples


def run_hipstr_pipeline(vcf_dir, out_prefix):
    print("\n=== Running HipSTR preprocessing ===")

    full_info = []
    full_matrix = []
    sample_names = None

    for chrom in range(1, 23):
        path = os.path.join(vcf_dir, f"hipstr_chr{chrom}.vcf.gz")
        print(f" → {path}")
        
        info_chr, rows_chr, samples_chr = process_hipstr_chr(path)
        if sample_names is None and samples_chr is not None:
            sample_names = samples_chr

        full_info.extend(info_chr)
        full_matrix.extend(rows_chr)

    df_info = pd.DataFrame(full_info)
    df_matrix = pd.DataFrame(full_matrix, columns=sample_names)

    df_info.to_csv(f"{out_prefix}_str_info.tsv", sep="\t", index=False)
    df_matrix.to_csv(f"{out_prefix}_matrix.tsv", sep="\t")

    print("\nHipSTR preprocessing complete.\n")


# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess EnsembleTR or HipSTR VCFs")
    p.add_argument("--caller", required=True, choices=["ensembletr", "hipstr"],
                   help="Choose which STR caller to process (no both).")
    p.add_argument("--vcf-dir", required=True,
                   help="Directory with VCF files.")
    p.add_argument("--sample-type",
                   help="Sample type file (required for EnsembleTR).")
    p.add_argument("--out-prefix", required=True,
                   help="Prefix for output TSV files.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.caller == "ensembletr":
        if args.sample_type is None:
            raise ValueError("--sample-type must be provided for EnsembleTR.")
        run_ensembletr_pipeline(args.vcf_dir, args.sample_type, args.out_prefix)

    elif args.caller == "hipstr":
        run_hipstr_pipeline(args.vcf_dir, args.out_prefix)


if __name__ == "__main__":
    main()
