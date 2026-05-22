#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simulation, dNMF, evaluation, Q recovery, and plotting from a grid CSV."
    )
    parser.add_argument(
        "--grid",
        default="simulation_grid.csv",
        help="Simulation grid CSV.",
    )
    parser.add_argument(
        "--scripts-dir",
        default="scripts",
        help="Directory containing simulate.py, dNMF.py, eval.py, recover_q.py, and plot_admixture.py.",
    )
    parser.add_argument(
        "--simulated-dir",
        default="simulated_data",
        help="Directory for simulated STR matrices and truth files.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for per-setting pipeline results.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=8,
        help="Maximum K passed to dNMF.py.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of dNMF repeats per setting.",
    )
    parser.add_argument(
        "--sim-seed",
        type=int,
        default=123,
        help="Seed passed to simulate.py.",
    )
    parser.add_argument(
        "--dnmf-seed",
        type=int,
        default=42,
        help="Seed passed to dNMF.py and recover_q.py.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum mean avercorr required by eval.py.",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.1,
        help="Minimum adjacent avercorr drop required by eval.py.",
    )
    parser.add_argument(
        "--pop-gap",
        type=float,
        default=2.0,
        help="Population gap width passed to plot_admixture.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun steps even if expected outputs already exist.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=None,
        help=(
            "Optional simulation replicate number. In replicate mode, outputs are "
            "written to simulated_data/<setting_id>_repN and results/<setting_id>/repN."
        ),
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        help="Optional setting_id value(s) to run. Default: run all settings in the grid.",
    )
    parser.add_argument(
        "--grid-output",
        default=None,
        help=(
            "CSV path for the grid with selected_K, mean_rmse_pos_Q, and "
            "mean_rmse_neg_Q columns. Default: overwrite --grid."
        ),
    )
    parser.add_argument(
        "--no-grid-update",
        action="store_true",
        help="Do not write selected_K/RMSE columns back to the grid CSV.",
    )
    return parser.parse_args()


def is_used(value):
    if value is None:
        return False
    value = str(value).strip()
    return value != "" and value != "-"


def run_command(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command:\n")
        log.write(" ".join(str(x) for x in cmd))
        log.write("\n\n")
        log.flush()

        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}. See log: {log_path}"
        )


def simulate_cmd(row, simulate_py, out_prefix, sim_seed):
    required_columns = [
        "scenario",
        "mode",
        "k",
        "n_individuals",
        "n_loci",
        "sigma",
    ]
    missing_required = [column for column in required_columns if not is_used(row.get(column))]
    if missing_required:
        raise ValueError(
            f"Setting {row.get('setting_id', '<unknown>')} has missing required "
            f"simulation column(s): {', '.join(missing_required)}"
        )

    cmd = [
        sys.executable,
        str(simulate_py),
        "--scenario",
        row["scenario"],
        "--mode",
        row["mode"],
        "--k",
        row["k"],
        "--n_individuals",
        row["n_individuals"],
        "--n_loci",
        row["n_loci"],
        "--sigma",
        row["sigma"],
        "--seed",
        str(sim_seed),
        "--out_prefix",
        str(out_prefix),
    ]

    optional_args = {
        "delta": "--delta",
        "alpha": "--alpha",
        "informative_fraction": "--informative_fraction",
        "direction_bias_fraction": "--direction_bias_fraction",
        "min_admixed_pops": "--min-admixed-pops",
        "max_admixed_pops": "--max-admixed-pops",
    }
    for column, flag in optional_args.items():
        if is_used(row.get(column)):
            cmd.extend([flag, row[column]])

    return cmd


def write_row_metadata(row, out_path):
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def read_rmse_summary(path):
    if not path.exists():
        return "-", "-"

    summary = pd.read_csv(path, sep="\t")
    pos = summary.loc[summary["channel"] == "pos", "mean_individual_rmse"]
    neg = summary.loc[summary["channel"] == "neg", "mean_individual_rmse"]

    pos_value = f"{float(pos.iloc[0]):.6g}" if not pos.empty else "-"
    neg_value = f"{float(neg.iloc[0]):.6g}" if not neg.empty else "-"
    return pos_value, neg_value


def result_suffix(replicate):
    if replicate is None:
        return ""
    return f"_rep{replicate}"


def result_subdir(results_dir, setting_id, replicate):
    setting_dir = results_dir / setting_id
    if replicate is None:
        return setting_dir
    return setting_dir / f"rep{replicate}"


def main():
    args = parse_args()
    if args.replicate is not None and args.replicate < 1:
        raise ValueError("--replicate must be a positive integer.")

    root = Path.cwd()
    scripts_dir = root / args.scripts_dir
    simulated_dir = root / args.simulated_dir
    results_dir = root / args.results_dir

    simulate_py = scripts_dir / "simulate.py"
    dnmf_py = scripts_dir / "dNMF.py"
    eval_py = scripts_dir / "eval.py"
    recover_q_py = scripts_dir / "recover_q.py"
    plot_admixture_py = scripts_dir / "plot_admixture.py"

    grid_path = Path(args.grid)
    full_grid = pd.read_csv(grid_path, dtype=str, keep_default_na=False)
    full_grid = full_grid.replace("", "-")
    for column in ["selected_K", "mean_rmse_pos_Q", "mean_rmse_neg_Q"]:
        if column not in full_grid.columns:
            full_grid[column] = "-"

    grid = full_grid.copy()
    if args.settings is not None:
        requested = set(args.settings)
        available = set(grid["setting_id"])
        missing = sorted(requested - available)
        if missing:
            raise ValueError(
                "Requested setting_id value(s) not found in grid: "
                + ", ".join(missing)
            )
        grid = grid.loc[grid["setting_id"].isin(requested)].reset_index(drop=True)

    pipeline_rows = []
    suffix = result_suffix(args.replicate)
    sim_seed = args.sim_seed if args.replicate is None else args.sim_seed + args.replicate - 1
    dnmf_seed = args.dnmf_seed if args.replicate is None else args.dnmf_seed + args.replicate - 1

    simulated_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for _, row_series in grid.iterrows():
        row = row_series.to_dict()
        setting_id = row["setting_id"]
        label = setting_id if args.replicate is None else f"{setting_id} rep{args.replicate}"
        print(f"\n=== {label} ===")

        setting_results_dir = result_subdir(results_dir, setting_id, args.replicate)
        setting_results_dir.mkdir(parents=True, exist_ok=True)
        write_row_metadata(row, setting_results_dir / "setting.csv")

        sim_prefix = simulated_dir / f"{setting_id}{suffix}"
        str_matrix = Path(f"{sim_prefix}.D.tsv")
        true_q = Path(f"{sim_prefix}.true_Q.tsv")
        metadata = Path(f"{sim_prefix}.metadata.tsv")

        if args.force or not str_matrix.exists():
            print("Running simulation")
            run_command(
                simulate_cmd(row, simulate_py, sim_prefix, sim_seed),
                setting_results_dir / "simulate.log",
            )
        else:
            print("Simulation exists; skipping")

        dnmf_prefix = setting_results_dir / f"{setting_id}_dnmf"
        dnmf_csv = Path(f"{dnmf_prefix}.csv")
        if args.force or not dnmf_csv.exists():
            print("Running dNMF")
            run_command(
                [
                    sys.executable,
                    str(dnmf_py),
                    "-f",
                    str(str_matrix),
                    "-o",
                    str(dnmf_prefix),
                    "-k",
                    str(args.max_k),
                    "--n-runs",
                    str(args.n_runs),
                    "--seed",
                    str(dnmf_seed),
                ],
                setting_results_dir / "dnmf.log",
            )
        else:
            print("dNMF result exists; skipping")

        best_k_csv = setting_results_dir / "best_k.csv"
        summary_csv = setting_results_dir / "k_summary.csv"
        if args.force or not summary_csv.exists():
            print("Evaluating K")
            run_command(
                [
                    sys.executable,
                    str(eval_py),
                    "-f",
                    str(dnmf_csv),
                    "-o",
                    str(best_k_csv),
                    "--summary-output",
                    str(summary_csv),
                    "--threshold",
                    str(args.threshold),
                    "--drop-threshold",
                    str(args.drop_threshold),
                ],
                setting_results_dir / "eval.log",
            )
        else:
            print("Evaluation exists; skipping")

        best_k = pd.read_csv(best_k_csv) if best_k_csv.exists() else pd.DataFrame()
        selected = not best_k.empty
        selected_k = "-"
        mean_rmse_pos = "-"
        mean_rmse_neg = "-"

        status = "selected_k" if selected else "no_selected_k"
        (setting_results_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
        print(f"Evaluation status: {status}")

        if selected:
            selected_k = str(int(best_k.iloc[0]["K"]))
            recovered_prefix = setting_results_dir / f"{setting_id}_recovered"
            q_pos = Path(f"{recovered_prefix}.Q_pos.tsv")
            q_neg = Path(f"{recovered_prefix}.Q_neg.tsv")
            pos_mapping = Path(f"{recovered_prefix}.pos_true_component_mapping.tsv")
            neg_mapping = Path(f"{recovered_prefix}.neg_true_component_mapping.tsv")

            if args.force or not q_pos.exists() or not q_neg.exists():
                print("Recovering Q")
                run_command(
                    [
                        sys.executable,
                        str(recover_q_py),
                        "--dnmf-result",
                        str(dnmf_csv),
                        "--str-matrix",
                        str(str_matrix),
                        "--true-q",
                        str(true_q),
                        "-o",
                        str(recovered_prefix),
                        "--threshold",
                        str(args.threshold),
                        "--drop-threshold",
                        str(args.drop_threshold),
                        "--seed",
                        str(dnmf_seed),
                    ],
                    setting_results_dir / "recover_q.log",
                )
            else:
                print("Recovered Q exists; skipping")

            rmse_summary = Path(f"{recovered_prefix}.true_q_rmse_summary.tsv")
            mean_rmse_pos, mean_rmse_neg = read_rmse_summary(rmse_summary)

            plot_png = setting_results_dir / f"{setting_id}_admixture.png"
            if args.force or not plot_png.exists():
                print("Plotting admixture")
                run_command(
                    [
                        sys.executable,
                        str(plot_admixture_py),
                        "--true-q",
                        str(true_q),
                        "--q-pos",
                        str(q_pos),
                        "--q-neg",
                        str(q_neg),
                        "--pos-mapping",
                        str(pos_mapping),
                        "--neg-mapping",
                        str(neg_mapping),
                        "--metadata",
                        str(metadata),
                        "--pop-gap",
                        str(args.pop_gap),
                        "-o",
                        str(plot_png),
                    ],
                    setting_results_dir / "plot_admixture.log",
                )
            else:
                print("Admixture plot exists; skipping")

        pipeline_rows.append(
            {
                "setting_id": setting_id,
                "replicate": args.replicate if args.replicate is not None else "-",
                "status": status,
                "selected_K": selected_k,
                "mean_rmse_pos_Q": mean_rmse_pos,
                "mean_rmse_neg_Q": mean_rmse_neg,
                "sim_prefix": str(sim_prefix),
                "result_dir": str(setting_results_dir),
                "dnmf_result": str(dnmf_csv),
                "best_k_file": str(best_k_csv),
                "summary_file": str(summary_csv),
            }
        )

        row_mask = full_grid["setting_id"] == setting_id
        if not args.no_grid_update and args.replicate is None:
            full_grid.loc[row_mask, "selected_K"] = selected_k
            full_grid.loc[row_mask, "mean_rmse_pos_Q"] = mean_rmse_pos
            full_grid.loc[row_mask, "mean_rmse_neg_Q"] = mean_rmse_neg

    pipeline_summary = pd.DataFrame(pipeline_rows)
    summary_name = (
        "pipeline_summary.csv"
        if args.replicate is None
        else f"pipeline_summary_rep{args.replicate}.csv"
    )
    pipeline_summary.to_csv(results_dir / summary_name, index=False)
    print(f"\nSaved pipeline summary to: {results_dir / summary_name}")

    should_update_grid = not args.no_grid_update and args.replicate is None
    if should_update_grid or args.grid_output is not None:
        grid_output = Path(args.grid_output) if args.grid_output is not None else grid_path
        full_grid.to_csv(grid_output, index=False)
        print(f"Saved updated grid to: {grid_output}")
    else:
        print("Skipped grid update.")


if __name__ == "__main__":
    main()
