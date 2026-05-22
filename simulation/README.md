# Simulation Benchmark

This folder contains a simulation workflow for evaluating dNMF on synthetic STR difference matrices. The benchmark generates simulated data, runs dNMF, selects the best K, recovers ancestry proportions when possible, plots admixture results, and summarizes replicate-level performance.

## Repository Layout

```text
simulation/
  sim_rep.csv              # final simulation design, one row per setting
  run_pipeline.py          # runs simulation -> dNMF -> K evaluation -> Q recovery -> plotting
  collect_results.py       # collects replicate results after local or Slurm runs
  README.md

  scripts/
    simulate.py            # generates D, true_Q, true_M, and metadata
    dNMF.py                # runs positive/negative-channel NMF across K
    eval.py                # selects best K from dNMF stability results
    recover_q.py           # recovers Q_pos and Q_neg at selected K and compares to true Q
    plot_admixture.py      # plots true Q, recovered Q_pos, and recovered Q_neg

  simulated_data/          # generated simulation files
  results/                 # dNMF, evaluation, recovery, plots, and summaries
```

## Simulation Model

The simulator generates:

```text
D = Q @ M + noise
```

where:

- `Q` is the true individual ancestry proportion matrix.
- `M` is the population-specific STR effect matrix.
- `D` is the simulated STR matrix.
- `noise` is Gaussian noise controlled by `sigma`.

The output STR matrix is written as:

```text
STRs x samples
```

This orientation is expected by `scripts/dNMF.py`.

## Scenarios

| Scenario | Meaning | Main parameters |
|---|---|---|
| `null` | No ancestry-informative STR effects; `D` is noise-driven. | `sigma` |
| `symmetric` | A fraction of loci have ancestry-informative effects in both directions. | `informative_fraction`, `delta`, `sigma` |
| `expansion_only` | A fraction of loci have positive-only ancestry effects. | `informative_fraction`, `delta`, `sigma` |
| `mixed` | Symmetric ancestry-informative loci plus additional direction-biased loci. | `informative_fraction`, `direction_bias_fraction`, `delta`, `sigma` |

## Ancestry Modes

| Mode | Meaning |
|---|---|
| `discrete` | Each individual belongs to one population. |
| `admixed` | Each individual has sparse ancestry from 2-4 populations, including a main population. The main population is constrained to have the largest ancestry proportion. |

For admixed simulations, `alpha` controls the Dirichlet concentration. Lower values give more uneven ancestry proportions; higher values give more even admixture among the selected populations.

## Simulation Design

The main design file is [sim_rep.csv](sim_rep.csv). It contains 17 unduplicated settings. Replicates are handled by `run_pipeline.py`, so the design file keeps one row per setting.

| setting_id | scenario | mode | k | delta | sigma | alpha | informative_fraction | direction_bias_fraction | purpose |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `null_discrete` | `null` | `discrete` | 5 | - | 1 | - | - | - | False-positive baseline with discrete populations |
| `null_admixed_alpha05` | `null` | `admixed` | 5 | - | 1 | 0.5 | - | - | False-positive baseline with sparse admixed ancestry |
| `sym_standard` | `symmetric` | `discrete` | 5 | 1 | 1 | - | 0.1 | - | Standard balanced ancestry signal |
| `sym_admixed_alpha05` | `symmetric` | `admixed` | 5 | 1 | 1 | 0.5 | 0.1 | - | Standard balanced ancestry signal under sparse admixed ancestry |
| `sym_admixed_alpha05_dense` | `symmetric` | `admixed` | 5 | 1 | 1 | 0.5 | 0.15 | - | Denser symmetric admixed signal |
| `exp_standard` | `expansion_only` | `discrete` | 5 | 1 | 1 | - | 0.1 | - | Standard positive-only expansion signal |
| `exp_admixed_alpha05` | `expansion_only` | `admixed` | 5 | 1 | 1 | 0.5 | 0.1 | - | Standard positive-only expansion signal under sparse admixed ancestry |
| `mix_moderate` | `mixed` | `discrete` | 5 | 1 | 1 | - | 0.1 | 0.05 | Main mixed benchmark with discrete populations |
| `mix_admixed_alpha05_dir005` | `mixed` | `admixed` | 5 | 1 | 1 | 0.5 | 0.1 | 0.05 | Mixed admixed benchmark with low directional bias |
| `mix_admixed_alpha05_total015_dir010` | `mixed` | `admixed` | 5 | 1 | 1 | 0.5 | 0.05 | 0.1 | Fixed total signal 0.15 with stronger directional fraction |
| `mix_admixed_alpha05_dir010` | `mixed` | `admixed` | 5 | 1 | 1 | 0.5 | 0.1 | 0.1 | Mixed admixed benchmark with moderate directional bias |
| `mix_admixed_alpha05_dir020` | `mixed` | `admixed` | 5 | 1 | 1 | 0.5 | 0.1 | 0.2 | Mixed admixed benchmark with strong directional bias |
| `mix_admixed_alpha10_dir005` | `mixed` | `admixed` | 5 | 1 | 1 | 1 | 0.1 | 0.05 | Mixed admixed benchmark with more even sparse admixture |
| `mix_admixed_noisy_alpha05_dense` | `mixed` | `admixed` | 5 | 1 | 2 | 0.5 | 0.2 | 0.05 | Dense mixed admixed signal under high noise |
| `mix_admixed_noisy_alpha05` | `mixed` | `admixed` | 5 | 1 | 2 | 0.5 | 0.1 | 0.05 | Mixed admixed robustness setting under high noise |
| `mix_admixed_alpha05_dense` | `mixed` | `admixed` | 5 | 1 | 1 | 0.5 | 0.2 | 0.05 | Mixed admixed robustness setting with more informative loci |
| `mix_admixed_alpha10_dense` | `mixed` | `admixed` | 5 | 1 | 1 | 1 | 0.2 | 0.05 | More informative loci with more even sparse admixture |

All settings use `n_individuals=500` and `n_loci=5000`.

## Running One Setting

To run one setting locally:

```bash
python run_pipeline.py \
  --grid sim_rep.csv \
  --settings mix_admixed_alpha05_dir010 \
  --max-k 8 \
  --n-runs 5
```

This writes outputs under:

```text
simulated_data/<setting_id>.*
results/<setting_id>/
```

## Running Replicates

Use `--replicate N` to run one simulation replicate across the selected settings. In replicate mode, `run_pipeline.py` writes separate output paths and does not update the grid file.

```bash
python run_pipeline.py \
  --grid sim_rep.csv \
  --replicate 1 \
  --max-k 8 \
  --n-runs 5
```

Replicate output layout:

```text
simulated_data/<setting_id>_rep1.*
results/<setting_id>/rep1/
results/pipeline_summary_rep1.csv
```

Seeds are offset by replicate number:

```text
rep1: sim_seed=123, dnmf_seed=42
rep2: sim_seed=124, dnmf_seed=43
...
```

## Slurm Array Example

The 5-replicate benchmark can be run as a 5-task Slurm array, where each task runs one replicate across all 17 settings.
The current Slurm script is used for running on ZHAW earth cluster.


## Collecting Results

After the replicate jobs finish:

```bash
python collect_results.py \
  --grid sim_rep.csv \
  --results-dir results \
  --replicates 1 2 3 4 5
```

This writes:

```text
results/replicate_results.csv
sim_rep_results.csv
```

If a top-level `results/pipeline_summary_repN.csv` is missing, `collect_results.py` can reconstruct it from:

```text
results/<setting_id>/repN/
```

## Output Files

Each simulation prefix produces:

```text
<prefix>.D.tsv
<prefix>.true_Q.tsv
<prefix>.true_M.tsv
<prefix>.metadata.tsv
```

Each pipeline result folder contains:

```text
setting.csv
simulate.log
<setting_id>_dnmf.csv
dnmf.log
best_k.csv
k_summary.csv
eval.log
status.txt
```

If a valid K is selected, the folder also contains:

```text
<setting_id>_recovered.Q_pos.tsv
<setting_id>_recovered.Q_neg.tsv
<setting_id>_recovered.true_q_rmse_summary.tsv
<setting_id>_recovered.pos_true_component_mapping.tsv
<setting_id>_recovered.neg_true_component_mapping.tsv
<setting_id>_admixture.png
recover_q.log
plot_admixture.log
```

## K Selection

`scripts/eval.py` selects K using the current rule:

```text
Choose the maximum K before avercorr becomes unstable, requiring mean_avercorr > 0.9.
```

Operationally, `eval.py` summarizes dNMF results by K, detects the first adjacent stability drop above `--drop-threshold`, and selects the maximum eligible K before that instability. If there is no major drop, it selects the maximum K with `mean_avercorr > --threshold`.

Default values:

```text
threshold = 0.9
drop_threshold = 0.1
max_k = 8
n_runs = 5
```

## Result Summary Columns

`sim_rep_results.csv` includes the design columns plus:

| Column | Meaning |
|---|---|
| `n_replicates` | Number of collected simulation replicates. |
| `n_valid_K` | Number of replicates with any selected K. |
| `selection_rate` | `n_valid_K / n_replicates`. |
| `n_selected_true_K` | Number of replicates where selected K equals true K. |
| `true_K_selection_rate` | `n_selected_true_K / n_replicates`. |
| `selected_K_mode` | Most frequent selected K among valid replicates. |
| `mean_selected_K` | Mean selected K among valid replicates. |
| `mean_rmse_pos_Q` | Mean individual-level RMSE for recovered positive-channel Q. |
| `sd_rmse_pos_Q` | Standard deviation of positive-channel RMSE. |
| `mean_rmse_neg_Q` | Mean individual-level RMSE for recovered negative-channel Q. |
| `sd_rmse_neg_Q` | Standard deviation of negative-channel RMSE. |

RMSE is only available when `recover_q.py` can compare recovered Q to true Q. If selected K differs from true K, RMSE is reported as `-` for that replicate.

## Current Five-Replicate Summary

The latest collected five-replicate summary is in [sim_rep_results.csv](sim_rep_results.csv). Broadly:

- Null and expansion-only settings selected no K across all replicates.
- `sym_standard` and `mix_moderate` recovered true K in all replicates.
- Sparse admixed symmetric and low-direction mixed settings mostly failed the current stability threshold.
- `mix_admixed_alpha05_dir010`, `mix_admixed_alpha05_dir020`, and `mix_admixed_alpha05_dense` selected true K in all replicates.
- `mix_admixed_alpha10_dense` selected a valid K in all replicates, but one replicate selected `K=3` instead of the true `K=5`.
