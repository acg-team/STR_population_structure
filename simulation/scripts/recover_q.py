#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

from dNMF import split_df
from eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recover dNMF Q at the selected best K and compare it with "
            "simulation truth from true_Q.tsv."
        )
    )
    parser.add_argument(
        "--dnmf-result",
        required=True,
        help="dNMF result CSV used to select best K.",
    )
    parser.add_argument(
        "--str-matrix",
        required=True,
        help="Simulated STR matrix, rows = STRs and columns = samples.",
    )
    parser.add_argument(
        "--true-q",
        required=True,
        help="Simulation true_Q.tsv file.",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        required=True,
        help="Output prefix for recovered Q and RMSE files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum mean avercorr required for a K to be eligible.",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.1,
        help="Minimum adjacent decrease in mean avercorr to call instability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for recovering Q at best K.",
    )
    return parser.parse_args()


def normalize_rows(x):
    denom = np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    return x / denom


def fit_q(matrix, k, seed):
    model = NMF(
        n_components=k,
        init="nndsvd",
        solver="cd",
        max_iter=10000,
        random_state=seed,
    )
    q = model.fit_transform(matrix)
    return normalize_rows(q), model


def align_to_true(q_est, q_true):
    if q_est.shape[1] != q_true.shape[1]:
        raise ValueError(
            "Recovered Q and true_Q have different numbers of components: "
            f"{q_est.shape[1]} vs {q_true.shape[1]}"
        )

    corr = np.corrcoef(q_est.T, q_true.T)[: q_est.shape[1], q_est.shape[1] :]
    est_ind, true_ind = linear_sum_assignment(-np.abs(corr))
    matched_corr = corr[est_ind, true_ind]

    q_aligned = np.zeros_like(q_est)
    q_aligned[:, true_ind] = q_est[:, est_ind]

    mapping = pd.DataFrame(
        {
            "estimated_component": est_ind + 1,
            "true_component": true_ind + 1,
            "corr": matched_corr,
            "abs_corr": np.abs(matched_corr),
        }
    )

    return q_aligned, corr, mapping


def rmse_to_true(q_est, q_true, label):
    q_aligned, corr, mapping = align_to_true(q_est, q_true)
    per_individual_rmse = np.sqrt(np.mean((q_aligned - q_true) ** 2, axis=1))
    per_component_rmse = np.sqrt(np.mean((q_aligned - q_true) ** 2, axis=0))

    individual = pd.DataFrame(
        {
            "individual_index": np.arange(q_true.shape[0]),
            f"{label}_rmse": per_individual_rmse,
        }
    )

    component = pd.DataFrame(
        {
            "true_component": np.arange(1, q_true.shape[1] + 1),
            f"{label}_rmse": per_component_rmse,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "channel": label,
                "mean_individual_rmse": per_individual_rmse.mean(),
                "median_individual_rmse": np.median(per_individual_rmse),
                "min_individual_rmse": per_individual_rmse.min(),
                "max_individual_rmse": per_individual_rmse.max(),
                "overall_rmse": np.sqrt(np.mean((q_aligned - q_true) ** 2)),
                "mean_abs_component_corr": mapping["abs_corr"].mean(),
            }
        ]
    )

    return q_aligned, corr, mapping, individual, component, summary


def main():
    args = parse_args()

    best, _, reason = evaluate(
        args.dnmf_result,
        threshold=args.threshold,
        drop_threshold=args.drop_threshold,
    )
    if best.empty:
        raise ValueError("No best K was selected from the dNMF result file.")

    best_k = int(best.iloc[0]["K"])
    print(f"Selected best K = {best_k}")
    print(reason)

    x_pos, x_neg = split_df(args.str_matrix)
    q_pos, _ = fit_q(x_pos, best_k, args.seed)
    q_neg, _ = fit_q(x_neg, best_k, args.seed)

    q_true = pd.read_csv(args.true_q, sep="\t", header=None).to_numpy()
    if q_true.shape[0] != q_pos.shape[0]:
        raise ValueError(
            "true_Q and recovered Q have different numbers of samples: "
            f"{q_true.shape[0]} vs {q_pos.shape[0]}"
        )

    (
        q_pos_aligned,
        pos_true_corr,
        pos_mapping,
        pos_individual_rmse,
        pos_component_rmse,
        pos_summary,
    ) = rmse_to_true(q_pos, q_true, "pos")
    (
        q_neg_aligned,
        neg_true_corr,
        neg_mapping,
        neg_individual_rmse,
        neg_component_rmse,
        neg_summary,
    ) = rmse_to_true(q_neg, q_true, "neg")
    
    rmse_summary = pd.concat([pos_summary, neg_summary], ignore_index=True)

    out_prefix = Path(args.output_prefix)
    pd.DataFrame(q_pos).to_csv(f"{out_prefix}.Q_pos.tsv", sep="\t", index=False)
    pd.DataFrame(q_neg).to_csv(f"{out_prefix}.Q_neg.tsv", sep="\t", index=False)
    # pd.DataFrame(q_pos_aligned).to_csv(
    #     f"{out_prefix}.Q_pos_aligned_to_true.tsv", sep="\t", index=False
    # )
    # pd.DataFrame(q_neg_aligned).to_csv(
    #     f"{out_prefix}.Q_neg_aligned_to_true.tsv", sep="\t", index=False
    # )
    # pd.DataFrame(pos_true_corr).to_csv(
    #     f"{out_prefix}.pos_true_component_corr.tsv", sep="\t", index=False
    # )
    # pd.DataFrame(neg_true_corr).to_csv(
    #     f"{out_prefix}.neg_true_component_corr.tsv", sep="\t", index=False
    # )
    pos_mapping.to_csv(f"{out_prefix}.pos_true_component_mapping.tsv", sep="\t", index=False)
    neg_mapping.to_csv(f"{out_prefix}.neg_true_component_mapping.tsv", sep="\t", index=False)
    # pos_individual_rmse.to_csv(
    #     f"{out_prefix}.pos_individual_rmse.tsv", sep="\t", index=False
    # )
    # neg_individual_rmse.to_csv(
    #     f"{out_prefix}.neg_individual_rmse.tsv", sep="\t", index=False
    # )
    # pos_component_rmse.to_csv(
    #     f"{out_prefix}.pos_component_rmse.tsv", sep="\t", index=False
    # )
    # neg_component_rmse.to_csv(
    #     f"{out_prefix}.neg_component_rmse.tsv", sep="\t", index=False
    # )
    rmse_summary.to_csv(f"{out_prefix}.true_q_rmse_summary.tsv", sep="\t", index=False)

    print("Q_pos versus true_Q component mapping:")
    print(pos_mapping.to_string(index=False))
    print()
    print("Q_neg versus true_Q component mapping:")
    print(neg_mapping.to_string(index=False))
    print()
    print("Per-individual RMSE summary versus true_Q:")
    print(rmse_summary.to_string(index=False))


if __name__ == "__main__":
    main()
