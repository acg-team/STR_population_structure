#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect replicate-level pipeline summaries into setting-level results."
    )
    parser.add_argument(
        "--grid",
        default="sim_rep.csv",
        help="Input simulation grid with one row per setting.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing pipeline_summary_repN.csv files.",
    )
    parser.add_argument(
        "--replicate-output",
        default=None,
        help="Output CSV for all replicate-level rows. Default: results/replicate_results.csv.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Output CSV for setting-level summary. Default: sim_rep_results.csv.",
    )
    parser.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Replicate numbers to collect. Missing pipeline_summary_repN.csv files "
            "are reconstructed from results/<setting_id>/repN when possible."
        ),
    )
    parser.add_argument(
        "--true-k-column",
        default="k",
        help="Grid column containing the true K.",
    )
    return parser.parse_args()


def numeric_or_nan(series):
    return pd.to_numeric(series.replace("-", pd.NA), errors="coerce")


def mode_or_dash(series):
    values = series.dropna()
    values = values[values.astype(str) != "-"]
    if values.empty:
        return "-"
    return str(values.mode().iloc[0])


def fmt(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def read_rmse_summary(path):
    if not path.exists():
        return "-", "-"

    summary = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    pos = summary.loc[summary["channel"] == "pos", "mean_individual_rmse"]
    neg = summary.loc[summary["channel"] == "neg", "mean_individual_rmse"]
    return (
        f"{float(pos.iloc[0]):.6g}" if not pos.empty else "-",
        f"{float(neg.iloc[0]):.6g}" if not neg.empty else "-",
    )


def reconstruct_replicate_summary(grid, results_dir, replicate):
    rows = []
    for _, grid_row in grid.iterrows():
        setting_id = grid_row["setting_id"]
        result_dir = results_dir / setting_id / f"rep{replicate}"
        best_k_file = result_dir / "best_k.csv"
        summary_file = result_dir / "k_summary.csv"
        dnmf_result = result_dir / f"{setting_id}_dnmf.csv"
        sim_prefix = Path("simulated_data") / f"{setting_id}_rep{replicate}"

        selected_k = "-"
        mean_rmse_pos = "-"
        mean_rmse_neg = "-"
        status = "missing"

        status_file = result_dir / "status.txt"
        if status_file.exists():
            status = status_file.read_text(encoding="utf-8").strip()

        if best_k_file.exists() and best_k_file.stat().st_size > 0:
            best_k = pd.read_csv(best_k_file, dtype=str, keep_default_na=False)
            if not best_k.empty and "K" in best_k.columns:
                selected_k = str(int(float(best_k.iloc[0]["K"])))
                status = "selected_k"
                rmse_file = result_dir / f"{setting_id}_recovered.true_q_rmse_summary.tsv"
                mean_rmse_pos, mean_rmse_neg = read_rmse_summary(rmse_file)
        elif status == "missing" and result_dir.exists():
            status = "no_selected_k"

        rows.append(
            {
                "setting_id": setting_id,
                "replicate": replicate,
                "status": status,
                "selected_K": selected_k,
                "mean_rmse_pos_Q": mean_rmse_pos,
                "mean_rmse_neg_Q": mean_rmse_neg,
                "sim_prefix": str(sim_prefix),
                "result_dir": str(result_dir),
                "dnmf_result": str(dnmf_result),
                "best_k_file": str(best_k_file),
                "summary_file": str(summary_file),
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    grid_path = Path(args.grid)
    results_dir = Path(args.results_dir)
    replicate_output = (
        Path(args.replicate_output)
        if args.replicate_output is not None
        else results_dir / "replicate_results.csv"
    )
    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else Path("sim_rep_results.csv")
    )

    grid = pd.read_csv(grid_path, dtype=str, keep_default_na=False).replace("", "-")

    available_summary_files = {
        int(path.stem.replace("pipeline_summary_rep", "")): path
        for path in results_dir.glob("pipeline_summary_rep*.csv")
        if path.stem.replace("pipeline_summary_rep", "").isdigit()
    }
    target_replicates = (
        args.replicates
        if args.replicates is not None
        else sorted(available_summary_files)
    )
    if not target_replicates:
        raise FileNotFoundError(
            f"No pipeline_summary_rep*.csv files found in {results_dir}."
        )

    replicate_tables = []
    reconstructed = []
    for replicate in target_replicates:
        path = available_summary_files.get(replicate)
        if path is not None:
            table = pd.read_csv(path, dtype=str, keep_default_na=False).replace("", "-")
            if "replicate" not in table.columns:
                table["replicate"] = replicate
        else:
            table = reconstruct_replicate_summary(grid, results_dir, replicate)
            reconstructed.append(replicate)
            if (table["status"] == "missing").all():
                raise FileNotFoundError(
                    f"Could not find or reconstruct replicate {replicate}."
                )
            path = results_dir / f"pipeline_summary_rep{replicate}.csv"
            table.to_csv(path, index=False)
            print(f"Reconstructed missing replicate summary: {path}")
        if "replicate" not in table.columns:
            table["replicate"] = replicate
        replicate_tables.append(table)

    replicate_results = pd.concat(replicate_tables, ignore_index=True)
    replicate_results["selected_K_num"] = numeric_or_nan(replicate_results["selected_K"])
    replicate_results["mean_rmse_pos_Q_num"] = numeric_or_nan(
        replicate_results["mean_rmse_pos_Q"]
    )
    replicate_results["mean_rmse_neg_Q_num"] = numeric_or_nan(
        replicate_results["mean_rmse_neg_Q"]
    )
    replicate_results["has_selected_K"] = replicate_results["selected_K_num"].notna()

    if args.true_k_column not in grid.columns:
        raise ValueError(f"Grid does not contain true K column: {args.true_k_column}")

    true_k = grid[["setting_id", args.true_k_column]].rename(
        columns={args.true_k_column: "true_K"}
    )
    replicate_results = replicate_results.merge(true_k, on="setting_id", how="left")
    replicate_results["true_K_num"] = numeric_or_nan(replicate_results["true_K"])
    replicate_results["selected_true_K"] = (
        replicate_results["selected_K_num"] == replicate_results["true_K_num"]
    )

    grouped = replicate_results.groupby("setting_id", sort=False)
    summary = grouped.agg(
        n_replicates=("replicate", "nunique"),
        n_valid_K=("has_selected_K", "sum"),
        n_selected_true_K=("selected_true_K", "sum"),
        selected_K_mode=("selected_K", mode_or_dash),
        mean_selected_K=("selected_K_num", "mean"),
        mean_rmse_pos_Q=("mean_rmse_pos_Q_num", "mean"),
        sd_rmse_pos_Q=("mean_rmse_pos_Q_num", "std"),
        mean_rmse_neg_Q=("mean_rmse_neg_Q_num", "mean"),
        sd_rmse_neg_Q=("mean_rmse_neg_Q_num", "std"),
    ).reset_index()

    summary["selection_rate"] = summary["n_valid_K"] / summary["n_replicates"]
    summary["true_K_selection_rate"] = (
        summary["n_selected_true_K"] / summary["n_replicates"]
    )

    ordered_columns = [
        "setting_id",
        "n_replicates",
        "n_valid_K",
        "selection_rate",
        "n_selected_true_K",
        "true_K_selection_rate",
        "selected_K_mode",
        "mean_selected_K",
        "mean_rmse_pos_Q",
        "sd_rmse_pos_Q",
        "mean_rmse_neg_Q",
        "sd_rmse_neg_Q",
    ]
    summary = summary[ordered_columns]

    for column in summary.columns:
        if column != "setting_id":
            summary[column] = summary[column].map(fmt)

    final = grid.merge(summary, on="setting_id", how="left")
    final = final.fillna("-")

    replicate_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    replicate_results.drop(
        columns=[
            "selected_K_num",
            "mean_rmse_pos_Q_num",
            "mean_rmse_neg_Q_num",
            "has_selected_K",
            "true_K_num",
            "selected_true_K",
        ],
        errors="ignore",
    ).to_csv(replicate_output, index=False)
    final.to_csv(summary_output, index=False)

    print(f"Collected {len(target_replicates)} replicate(s).")
    if reconstructed:
        print("Reconstructed replicate summary file(s): " + ", ".join(map(str, reconstructed)))
    print(f"Saved replicate-level results to: {replicate_output}")
    print(f"Saved setting-level summary to: {summary_output}")


if __name__ == "__main__":
    main()
