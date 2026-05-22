#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select the best dNMF K as the maximum K before avercorr becomes "
            "unstable, requiring avercorr above a threshold."
        )
    )
    parser.add_argument(
        "-f", "--file-path", required=True, help="Path to dNMF result CSV."
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default=None,
        help="Optional output CSV for the selected best K summary.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional output CSV for the per-K summary table.",
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
    return parser.parse_args()


def summarize_by_k(df):
    if "repN" in df.columns:
        summary = (
            df.groupby("K", as_index=False)
            .agg(
                mean_avercorr=("avercorr", "mean"),
                sd_avercorr=("avercorr", "std"),
                mean_varcorr=("varcorr", "mean"),
                mean_match=("match", "mean"),
                mean_pos_err=("pos_err", "mean"),
                mean_neg_err=("neg_err", "mean"),
                n_reps=("repN", "nunique"),
            )
        )
    else:
        summary = df.copy()
        summary = summary.rename(
            columns={
                "avercorr": "mean_avercorr",
                "varcorr": "mean_varcorr",
                "match": "mean_match",
                "pos_err": "mean_pos_err",
                "neg_err": "mean_neg_err",
            }
        )
        summary["sd_avercorr"] = pd.NA
        summary["n_reps"] = 1

    summary = summary.sort_values("K").reset_index(drop=True)
    summary["next_mean_avercorr"] = summary["mean_avercorr"].shift(-1)
    summary["drop_to_next_k"] = (
        summary["mean_avercorr"] - summary["next_mean_avercorr"]
    )
    return summary


def select_best_k(summary, threshold=0.9, drop_threshold=0.1):
    major_drops = summary.loc[
        (summary["mean_avercorr"] > threshold)
        & (summary["drop_to_next_k"] >= drop_threshold)
    ]

    if not major_drops.empty:
        first_drop_k = major_drops.iloc[0]["K"]
        eligible = summary.loc[
            (summary["K"] <= first_drop_k)
            & (summary["mean_avercorr"] > threshold)
        ]
        reason = (
            f"Selected maximum K before avercorr becomes unstable "
            f"(drop >= {drop_threshold})."
        )
    else:
        eligible = summary.loc[summary["mean_avercorr"] > threshold]
        reason = (
            "No avercorr instability found; selected maximum K with "
            f"mean_avercorr > {threshold}."
        )

    if eligible.empty:
        return eligible.copy(), f"No K found with mean_avercorr > {threshold}."

    best_k = eligible["K"].max()
    best = eligible.loc[eligible["K"] == best_k].copy()
    best["selection_reason"] = reason
    return best, reason


def evaluate(file_path, threshold=0.9, drop_threshold=0.1):
    df = pd.read_csv(file_path)

    required_cols = {"K", "avercorr"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing_cols)}"
        )

    df["K"] = pd.to_numeric(df["K"])
    df["avercorr"] = pd.to_numeric(df["avercorr"])

    summary = summarize_by_k(df)
    best, reason = select_best_k(
        summary, threshold=threshold, drop_threshold=drop_threshold
    )
    return best, summary, reason


def main():
    args = parse_args()
    best, summary, reason = evaluate(
        args.file_path,
        threshold=args.threshold,
        drop_threshold=args.drop_threshold,
    )

    print("Per-K stability summary:")
    print(summary.to_string(index=False))
    print()

    if best.empty:
        print(reason)
    else:
        print("Best K result:")
        print(best.to_string(index=False))

    if args.output_path is not None:
        out_path = Path(args.output_path)
        best.to_csv(out_path, index=False)
        print(f"Saved selected best K to: {out_path}")

    if args.summary_output is not None:
        summary_out = Path(args.summary_output)
        summary.to_csv(summary_out, index=False)
        print(f"Saved per-K summary to: {summary_out}")


if __name__ == "__main__":
    main()
