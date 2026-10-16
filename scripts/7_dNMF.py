#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import argparse

# python dNMF.py \
#   -f data/str_matrix.tsv \
#   --str-info-file data/str_info.tsv \
#   --period 3 \
#   -o results/nmf_period3 \
#   -k 20 \
#   --n-runs 5 \
#   --seed 42


def parse_cla():
    parser = argparse.ArgumentParser(
        description="Cross-validate NMF on positive/negative STR genotype matrices."
    )
    parser.add_argument(
        "-f", "--file-path", type=str, required=True,
        help="Path to STR genotype matrix (rows = STRs, cols = samples)."
    )
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="Output prefix (CSV will be written)."
    )
    parser.add_argument(
        "--n-runs", type=int, default=5,
        help="Number of repeated NMF runs per K."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for repeated NMF runs."
    )
    parser.add_argument(
        "-k", "--max-k", type=int, default=20,
        help="Maximum K (components) to test, inclusive (range starts at 3)."
    )
    parser.add_argument(
        "--str-info-file", type=str, default=None,
        help="Optional STR info file with columns 'chr', 'str_id', 'period', 'str_var', 'methods'."
    )
    parser.add_argument(
        "--period", type=int, default=None,
        help="Optional STR period to subset to (e.g. 2, 3, 4...)."
    )
    return parser.parse_args()


def cross_validate_nmf(
    pos_matrix,
    neg_matrix,
    max_k,
    n_repeats=1,
    matching=0.9,
    random_state=42,
    output_path=None
):
    """
    Run NMF on pos/neg matrices for K in [3, max_k].

    Returns
    -------
    df_out : pandas.DataFrame
        Combined NMF results with one row per repeat and K.
    """

    K_range = range(3, max_k + 1)
    rng = np.random.default_rng(random_state)
    results = []

    for repeat in range(n_repeats):
        rs = rng.integers(0, 1_000_000)
        print(f"Repeat {repeat + 1}/{n_repeats}, seed={rs}")

        for K in K_range:
            print(f"  K = {K}")

            model_pos = NMF(
                n_components=K,
                init="nndsvd",
                solver="cd",
                max_iter=10000,
                random_state=rs,
            )
            Q_pos = model_pos.fit_transform(pos_matrix)

            model_neg = NMF(
                n_components=K,
                init="nndsvd",
                solver="cd",
                max_iter=10000,
                random_state=rs,
            )
            Q_neg = model_neg.fit_transform(neg_matrix)

            Q_pos /= Q_pos.sum(axis=1, keepdims=True)
            Q_neg /= Q_neg.sum(axis=1, keepdims=True)

            corr_matrix = np.corrcoef(Q_pos.T, Q_neg.T)[:K, K:]
            #cost_matrix = -np.abs(corr_matrix)
            cost_matrix = -corr_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_corr = corr_matrix[row_ind, col_ind]

            results.append({
                "repN": repeat + 1,
                "K": K,
                "avercorr": total_corr.mean(),
                "varcorr": total_corr.var(),
                "match": int(np.sum(np.abs(total_corr) > matching)),
                "pos_err": float(model_pos.reconstruction_err_),
                "neg_err": float(model_neg.reconstruction_err_),
            })

    df_out = pd.DataFrame(results)
    out_path = output_path if output_path.endswith(".csv") else output_path + ".csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved combined NMF results to: {out_path}")

    return df_out


def split_df(str_file, str_info_file=None, period=None):
    """
    Load STR matrix, optionally subset by STR info & period,
    then standardize and split into positive/negative parts.

    Returns:
      X_pos, X_neg (both samples x STRs)
    """
    # rows = STRs, cols = samples
    str_data = pd.read_csv(str_file, sep="\t", index_col=0, low_memory=False)

    if str_info_file is not None:
        str_info = pd.read_csv(str_info_file, sep="\t").query("str_var > 2").copy()

        # keep only HipSTR calls (3rd field in 'methods' == 1)
        hipstr = np.array([int(i.split("|")[2]) for i in str_info["methods"]])
        str_info = str_info.iloc[hipstr == 1, :]

        str_info["str_uid"] = str_info["chr"] + "_" + str_info["str_id"].astype("str")

        if period is not None:
            keep_uids = str_info.loc[str_info["period"] == period, "str_uid"]
        else:
            keep_uids = str_info["str_uid"]

        # intersect with available rows just in case
        keep_uids = [u for u in keep_uids if u in str_data.index]
        input_data = str_data.loc[keep_uids]
    else:
        input_data = str_data

    # input_data: STRs x samples -> transpose to samples x STRs
    #imputed = SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(
    #    input_data.T)
    #X_data = StandardScaler().fit_transform(imputed)
    X_data = input_data.T.values
    X_pos = np.clip(X_data, a_min=0, a_max=None)
    X_neg = np.clip(-X_data, a_min=0, a_max=None)

    return X_pos, X_neg


def main():
    args = parse_cla()
    print(f"Using random_state={args.seed}")

    X_pos, X_neg = split_df(
        args.file_path,
        str_info_file=args.str_info_file,
        period=args.period,
    )

    res = cross_validate_nmf(
        pos_matrix=X_pos,
        neg_matrix=X_neg,
        max_k=args.max_k,
        n_repeats=args.n_runs,
        random_state=args.seed,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
