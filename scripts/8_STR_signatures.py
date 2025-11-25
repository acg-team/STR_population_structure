#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import fisher_exact
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------
# Component matching
# ---------------------------------------------------------

def match_components(Q_pos: pd.DataFrame, Q_neg: pd.DataFrame):
    """
    Match positive and negative components using the Hungarian algorithm
    based on correlations between rows of Q_pos and Q_neg (samples x K).

    Returns
    -------
    pos_ind : np.ndarray
        Indices of positive components (order in Q_pos.columns).
    neg_ind : np.ndarray
        Indices of matched negative components (order in Q_neg.columns).
    corr_vals : np.ndarray
        Correlation values of matched pairs.
    """
    K = Q_pos.shape[1]
    corr_matrix = np.corrcoef(Q_pos.T, Q_neg.T)[:K, K:]
    cost_matrix = -np.abs(corr_matrix)
    pos_ind, neg_ind = linear_sum_assignment(cost_matrix)
    corr_vals = corr_matrix[pos_ind, neg_ind]
    return pos_ind, neg_ind, corr_vals


# ---------------------------------------------------------
# Extract top loci sets per component
# ---------------------------------------------------------

def extract_str_signature_sets(hs_pos: pd.DataFrame,
                               hs_neg: pd.DataFrame,
                               pos_ind: np.ndarray,
                               neg_ind: np.ndarray,
                               top_frac: float = 0.05):
    """
    Given H_pos, H_neg (as DataFrames) and matched indices, extract:
      - per-component correlation (H-space) between pos/neg loadings
      - per-component top STR index sets for pos, neg, and union

    hs_pos, hs_neg should be:
      rows = STRs (in same order as 'data' matrix)
      cols = components

    Parameters
    ----------
    hs_pos, hs_neg : pandas.DataFrame
        H matrices as STRs x K (after transpose).
    pos_ind, neg_ind : np.ndarray
        Indices from linear_sum_assignment matching pos to neg components.
    top_frac : float
        Fraction of STRs to keep per component (e.g. 0.05 for top 5%).

    Returns
    -------
    df_corr : pandas.DataFrame
        Columns: ['component', 'corr'] (H-space Pearson r for each K).

    pos_sets, neg_sets, per_un : dict
        For each component: set of STR row indices (integer positions).

    pos_loci, neg_loci : np.ndarray
        Global union of loci indices across all components (optional).
    """
    # Reorder by matched indices
    hs_pos = hs_pos.iloc[:, pos_ind]
    hs_neg = hs_neg.iloc[:, neg_ind]

    # Give them consistent integer column names 0..K-1
    hs_pos.columns = range(hs_pos.shape[1])
    hs_neg.columns = range(hs_neg.shape[1])

    K = hs_pos.shape[1]

    # per-component correlation in H-space
    corr_vals = []
    for i in range(K):
        r = stats.pearsonr(hs_pos.iloc[:, i], hs_neg.iloc[:, i])
        corr_vals.append(r.statistic)

    df_corr = pd.DataFrame({
        "component": [f"Component_{i+1}" for i in range(K)],
        "corr": corr_vals,
    })

    # top loci
    top_k = max(1, round(hs_pos.shape[0] * top_frac))

    pos_sets = {
        f"Component_{i + 1}": set(
            np.argsort(hs_pos.iloc[:, i].values)[::-1][:top_k]
        )
        for i in range(K)
    }
    neg_sets = {
        f"Component_{i + 1}": set(
            np.argsort(hs_neg.iloc[:, i].values)[::-1][:top_k]
        )
        for i in range(K)
    }

    per_un = {
        f"Component_{i + 1}": pos_sets[f"Component_{i + 1}"].union(
            neg_sets[f"Component_{i + 1}"]
        )
        for i in range(K)
    }

    pos_loci = np.array(list(set().union(*pos_sets.values())))
    neg_loci = np.array(list(set().union(*neg_sets.values())))

    return df_corr, pos_sets, neg_sets, per_un, pos_loci, neg_loci


# ---------------------------------------------------------
# Enrichment test 
# ---------------------------------------------------------

def enrichment_test(str_info: pd.DataFrame,
                    data: pd.DataFrame,
                    sets: dict,
                    tar: str) -> pd.DataFrame:
    """
    Perform motif/feature enrichment test for each component-specific set of loci.

    Parameters
    ----------
    str_info : pandas.DataFrame
        STR annotation table. Must contain:
          - 'str_uid' (unique STR ID matching data.index)
          - column 'tar' (e.g. 'period', 'motif', etc.)

    data : pandas.DataFrame
        STR matrix with index = str_uid (rows = STRs, cols = samples).

    sets : dict
        Mapping 'Component_i' -> set of integer row indices
        (positions into data.index).

    tar : str
        Column name in str_info to test enrichment for
        (e.g. 'period').

    Returns
    -------
    df_sig : pandas.DataFrame
        Enrichment results with odds ratio, enrichment ratio,
        raw and FDR-adjusted p-values.
    """

    df_sig = pd.DataFrame(columns=["component", tar,
                                   "enrichment_ratio", "odds_ratio", "p_value"])

    # Background counts
    total_period_counts = str_info[tar].value_counts()
    total_str_count = len(str_info)

    for i in range(len(sets.keys())):
        comp_name = f"Component_{i + 1}"
        set_name = f"S{i + 1}"

        # indices of driver loci for this component
        c_loci = list(sets[comp_name])

        # subset STRs in driver set (map via data index, which should be str_uid)
        driver_str_info = str_info.loc[
            str_info["str_uid"].isin(data.index[c_loci])
        ]
        driver_period_counts = driver_str_info[tar].value_counts()
        driver_count = len(driver_str_info)

        # skip empty
        if driver_count == 0:
            continue

        for period, driver_count_m in driver_period_counts.items():
            # background count for this category in all STRs
            total_for_period = total_period_counts.loc[period]

            # 2x2 table:
            # A: STRs with category 'period' that ARE in driver set
            A = driver_count_m
            # B: STRs with category 'period' that are NOT in driver set
            B = total_for_period - A
            # C: Other categories that ARE in driver set
            C = driver_count - A
            # D: Other categories that are NOT in driver set
            D = (total_str_count - total_for_period) - C

            contingency_table = [[A, B], [C, D]]
            odds_ratio, p_value = fisher_exact(
                contingency_table, alternative="greater"
            )

            enrichment_ratio = (A / driver_count) / (total_for_period / total_str_count)

            df_sig = pd.concat(
                [
                    df_sig,
                    pd.DataFrame([{
                        "component": set_name,
                        tar: period,
                        "enrichment_ratio": enrichment_ratio,
                        "odds_ratio": odds_ratio,
                        "p_value": p_value,
                    }]),
                ],
                ignore_index=True,
            )

    # multiple testing correction
    if len(df_sig) > 0:
        p_values = df_sig["p_value"].values
        reject, p_adjusted, _, _ = multipletests(p_values, method="fdr_bh")
        df_sig["p_value_adjusted"] = p_adjusted
        df_sig["is_significant"] = reject
    else:
        df_sig["p_value_adjusted"] = []
        df_sig["is_significant"] = []

    return df_sig


# ---------------------------------------------------------
# CLI + main pipeline
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract STR signatures from NMF (pos/neg) and run enrichment."
    )
    parser.add_argument(
        "--str-info",
        required=True,
        help="STR info TSV/CSV with 'str_uid' and the target column (e.g. 'period').",
    )
    parser.add_argument(
        "--data-matrix",
        required=True,
        help="STR data matrix (TSV; rows = STRs, index = str_uid, cols = samples) used for NMF.",
    )
    parser.add_argument(
        "--q-pos",
        required=True,
        help="Q_pos CSV (samples x K).",
    )
    parser.add_argument(
        "--q-neg",
        required=True,
        help="Q_neg CSV (samples x K).",
    )
    parser.add_argument(
        "--h-pos",
        required=True,
        help="H_pos CSV (raw; will be reset_index(drop=True).T to STRs x K).",
    )
    parser.add_argument(
        "--h-neg",
        required=True,
        help="H_neg CSV (raw; will be reset_index(drop=True).T to STRs x K).",
    )
    parser.add_argument(
        "--target-col",
        required=True,
        help="Column in str_info to test enrichment for (e.g. 'period', 'motif').",
    )
    parser.add_argument(
        "--top-frac",
        type=float,
        default=0.05,
        help="Fraction of STRs per component to include as drivers (default: 0.05).",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load STR info
    if args.str_info.endswith(".csv"):
        str_info = pd.read_csv(args.str_info)
    else:
        str_info = pd.read_csv(args.str_info, sep="\t")

    if "str_uid" not in str_info.columns:
        raise ValueError("str_info must contain a 'str_uid' column.")

    if args.target_col not in str_info.columns:
        raise ValueError(f"Column '{args.target_col}' not found in str_info.")

    # Load STR data matrix (loci x samples)
    data = pd.read_csv(args.data_matrix, sep="\t", index_col=0)

    # Load Q matrices
    Q_pos = pd.read_csv(args.q_pos, index_col=0)
    Q_neg = pd.read_csv(args.q_neg, index_col=0)

    # Load H matrices and reshape to STRs x K
    H_pos_raw = pd.read_csv(args.h_pos, index_col=0)
    H_neg_raw = pd.read_csv(args.h_neg, index_col=0)

    H_pos = H_pos_raw.reset_index(drop=True).T
    H_neg = H_neg_raw.reset_index(drop=True).T

    # Make sure H rows align with data index (if needed)
    if H_pos.shape[0] != data.shape[0]:
        print("[WARN] Number of STRs in H_pos/H_neg does not match data_matrix rows.")
        print(f"  H_pos rows: {H_pos.shape[0]}, data_matrix rows: {data.shape[0]}")

    # 1) Match components (Q_pos / Q_neg)
    pos_ind, neg_ind, corr_vals_q = match_components(Q_pos, Q_neg)
    print("Component correlations in Q-space (pos vs neg):")
    print(corr_vals_q)

    # 2) Extract top loci sets per component (H-space)
    df_corr_h, pos_sets, neg_sets, per_un, pos_loci, neg_loci = \
        extract_str_signature_sets(H_pos, H_neg, pos_ind, neg_ind, top_frac=args.top_frac)

    # 3) Enrichment on union sets (per_un)
    df_enrich = enrichment_test(
        str_info=str_info,
        data=data,
        sets=per_un,
        tar=args.target_col,
    )

    # 4) Save outputs
    out_corr_q = args.out_prefix + "_Q_matching_corr.tsv"
    out_corr_h = args.out_prefix + "_H_matching_corr.tsv"
    out_enrich = args.out_prefix + "_enrichment.tsv"

    # Q-space corr meta
    df_corr_q = pd.DataFrame({
        "component": [f"Component_{i+1}" for i in range(len(corr_vals_q))],
        "corr_Q_pos_neg": corr_vals_q,
    })
    df_corr_q.to_csv(out_corr_q, sep="\t", index=False)

    df_corr_h.to_csv(out_corr_h, sep="\t", index=False)
    df_enrich.to_csv(out_enrich, sep="\t", index=False)

    print(f"Saved Q-space matching correlations to: {out_corr_q}")
    print(f"Saved H-space matching correlations to: {out_corr_h}")
    print(f"Saved enrichment results to: {out_enrich}")


if __name__ == "__main__":
    main()
