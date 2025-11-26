#!/usr/bin/env python3
"""
Geo-genetic Mantel test for SNP and STR distances (e.g. HGDP / 1KGP).

Inputs:
  - Individual-level STR distance matrix (samples × samples)
  - Individual-level SNP distance matrix (samples × samples)
  - Sample metadata with at least a 'Population' column
  - Population coordinates (e.g. hgdp_populations.tsv)

Pipeline:
  1. Aggregate individual-level STR/SNP distances to population-level
  2. Build population-level geographic distance matrix (Haversine)
  3. Perform Mantel tests:
       - STR vs geography
       - SNP vs geography
  4. Save results as a small TSV table with Mantel r and p
"""

import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from utils.distance_utils import mantel_test


# ================================================================
# Distance aggregation: individuals -> populations
# ================================================================

def build_population_distance_matrix(indiv_dist_df, sample_to_pop, agg="mean"):
    if isinstance(sample_to_pop, pd.DataFrame):
        if "Population" not in sample_to_pop.columns:
            raise ValueError("DataFrame 'sample_to_pop' must contain a 'Population' column.")
        pop_series = sample_to_pop["Population"]
    else:
        pop_series = sample_to_pop

    common_samples = indiv_dist_df.index.intersection(pop_series.index)
    if len(common_samples) == 0:
        raise ValueError("No overlapping samples between distance matrix and sample_to_pop.")

    D = indiv_dist_df.loc[common_samples, common_samples]
    pop_series = pop_series.loc[common_samples]

    pops = pop_series.unique().tolist()
    pop_dist = np.zeros((len(pops), len(pops)), dtype=float)

    for i, pi in enumerate(pops):
        idx_i = pop_series[pop_series == pi].index
        for j, pj in enumerate(pops):
            idx_j = pop_series[pop_series == pj].index

            sub = D.loc[idx_i, idx_j].values

            if pi == pj:
                if sub.shape[0] > 1:
                    mask = ~np.eye(sub.shape[0], dtype=bool)
                    vals = sub[mask]
                else:
                    vals = np.array([0.0])
            else:
                vals = sub.flatten()

            vals = vals[~np.isnan(vals)]
            pop_dist[i, j] = np.mean(vals) if len(vals) > 0 else np.nan

    return pd.DataFrame(pop_dist, index=pops, columns=pops)


# ================================================================
# Geography: haversine + pop distance matrix
# ================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def build_geo_dist_matrix(geo_tsv_path, restrict_pops=None):
    geo = pd.read_csv(geo_tsv_path, sep="\t")
    coords = geo[["Population name", "Population latitude", "Population longitude"]]
    coords.columns = ["pop", "lat", "lon"]

    if restrict_pops is not None:
        restrict_pops = set(restrict_pops)
        coords = coords[coords["pop"].isin(restrict_pops)].reset_index(drop=True)

    n = len(coords)
    geo_dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            geo_dist[i, j] = haversine(
                coords.lat.iloc[i], coords.lon.iloc[i],
                coords.lat.iloc[j], coords.lon.iloc[j],
            )

    return pd.DataFrame(geo_dist, index=coords["pop"], columns=coords["pop"])


# ================================================================
# Main pipeline
# ================================================================

def run_geo_mantel(
    str_dist_path,
    snp_dist_path,
    sample_metadata_path,
    geo_tsv_path,
    out_tsv_path,
    perms=10000,
):
    print(f"Loading STR distance matrix from: {str_dist_path}")
    str_dist_df = pd.read_csv(str_dist_path, sep="\t", index_col=0)

    print(f"Loading SNP distance matrix from: {snp_dist_path}")
    snp_dist_df = pd.read_csv(snp_dist_path, sep="\t", index_col=0)

    print(f"Loading sample metadata from: {sample_metadata_path}")
    sample_df = pd.read_csv(sample_metadata_path, sep="\t", index_col=0)
    if "Population" not in sample_df.columns:
        raise ValueError("sample_metadata must contain a 'Population' column.")

    sample_to_pop = sample_df["Population"]

    print("Building population-level STR matrix ...")
    str_pop_dist_df = build_population_distance_matrix(str_dist_df, sample_to_pop)

    print("Building population-level SNP matrix ...")
    snp_pop_dist_df = build_population_distance_matrix(snp_dist_df, sample_to_pop)

    common_pops = str_pop_dist_df.index.intersection(snp_pop_dist_df.index)
    print(f"Common populations: {len(common_pops)}")

    print("Building geo matrix ...")
    geo_dist_df = build_geo_dist_matrix(
        geo_tsv_path,
        restrict_pops=common_pops,
    )

    common_pops_final = common_pops.intersection(geo_dist_df.index)
    print(f"Pops used in Mantel test: {len(common_pops_final)}")

    str_gen = str_pop_dist_df.loc[common_pops_final, common_pops_final]
    snp_gen = snp_pop_dist_df.loc[common_pops_final, common_pops_final]
    geo = geo_dist_df.loc[common_pops_final, common_pops_final]

    print("Mantel test: STR vs geography ...")
    r_str, p_str = mantel_test(str_gen.values, geo.values, perms=perms, random_state=42)

    print("Mantel test: SNP vs geography ...")
    r_snp, p_snp = mantel_test(snp_gen.values, geo.values, perms=perms, random_state=43)

    results = pd.DataFrame([
        {
            "Marker": "STR",
            "n_populations": len(common_pops_final),
            "permutations": perms,
            "Mantel_r": r_str,
            "Mantel_p": p_str,
        },
        {
            "Marker": "SNP",
            "n_populations": len(common_pops_final),
            "permutations": perms,
            "Mantel_r": r_snp,
            "Mantel_p": p_snp,
        },
    ])

    print(f"Saving results to: {out_tsv_path}")
    results.to_csv(out_tsv_path, sep="\t", index=False)

    print("Done.")


# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Geo-genetic Mantel test for STR and SNP distances.")
    p.add_argument("--str-dist", required=True, help="Path to STR distance matrix (samples × samples).")
    p.add_argument("--snp-dist", required=True, help="Path to SNP distance matrix (samples × samples).")
    p.add_argument("--sample-metadata", required=True, help="TSV with 'Population' column.")
    p.add_argument("--geo-tsv", required=True, help="TSV with population coordinates.")
    p.add_argument("--out-tsv", required=True, help="Output TSV path for Mantel results.")
    p.add_argument("--perms", type=int, default=10000, help="Permutations (default 10000).")
    return p.parse_args()


def main():
    args = parse_args()
    run_geo_mantel(
        str_dist_path=args.str_dist,
        snp_dist_path=args.snp_dist,
        sample_metadata_path=args.sample_metadata,
        geo_tsv_path=args.geo_tsv,
        out_tsv_path=args.out_tsv,
        perms=args.perms,
    )


if __name__ == "__main__":
    main()
