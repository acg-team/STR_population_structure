#!/usr/bin/env python3
"""
Supervised population assignment from raw SNP/STR genotype matrices,
plus optional downsampling and per-chromosome RF accuracy analyses.

Input:
  - Raw genotype matrix (loci x samples), SNP or STR
  - Sample metadata with columns:
        'Superpopulation', 'Population'
    and index = sample IDs matching the matrix columns.
  - (Optional) chr map TSV for per-chrom analysis with columns:
        'locus', 'chr'
    where 'locus' matches the genotype matrix row index.

Main analyses:
  1) Supervised assignment (RF + NB) at:
       - Superpopulation (continental) level, or
       - Population (regional) level
     with optional PCA feature reduction.

Optional extras:
  2) Downsampling test: randomly (with replacement) sample subsets of loci,
     run RF on Superpopulation and Population labels, and track accuracy.
  3) Per-chromosome RF accuracy: run RF using only loci from each chromosome.

Outputs:
  - <out_prefix>_supervised_assignment.tsv
  - (optional) <out_prefix}_downsampling_rf.tsv
  - (optional) <out_prefix}_perchr_rf.tsv
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ================================================================
# Supervised assignment (RF + NB)
# ================================================================

def evaluate_supervised_assignment(
    X,
    sample_df,
    label_level="Superpopulation",   # or "Population"
    use_pca=False,
    n_pcs=50,
    n_splits=5,
    n_repeats=1,
    random_state=42,
):
    """
    Supervised population assignment using RF and NB,
    optionally with PCA in the pipeline.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix with shape (n_samples, n_features).
        Columns = SNP/STR features (or already PCs),
        index = sample IDs.

    sample_df : pandas.DataFrame
        Metadata with index = sample IDs, contains at least
        `label_level` (e.g. "Superpopulation" or "Population").

    label_level : str
        Column in sample_df used as the target.

    use_pca : bool
        If True, insert PCA before classifier.

    n_pcs : int
        Number of PCs to keep if use_pca=True.

    n_splits : int
        Number of CV folds (StratifiedKFold).

    n_repeats : int
        How many times to repeat CV with different random seeds.

    random_state : int
        Base random seed.

    Returns
    -------
    results_df : pandas.DataFrame
        One row per classifier, with mean/std accuracy.
    """

    # Align samples
    common_samples = X.index.intersection(sample_df.index)
    if len(common_samples) == 0:
        raise ValueError("No overlapping samples between genotype matrix and sample metadata.")

    X = X.loc[common_samples]
    y = sample_df.loc[common_samples, label_level]

    print(f"Using {len(common_samples)} samples for label='{label_level}'.")

    # Core preprocessing steps
    def build_pipeline(clf):
        steps = [
            ("imp", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler()),
        ]
        if use_pca:
            steps.append(("pca", PCA(n_components=n_pcs, random_state=random_state)))
        steps.append(("clf", clf))
        return Pipeline(steps)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=50,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    nb = GaussianNB()

    pipelines = {
        "RF": build_pipeline(rf),
        "NB": build_pipeline(nb),
    }

    rows = []

    for name, pipe in pipelines.items():
        all_scores = []

        for r in range(n_repeats):
            seed = random_state + r
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            scores = cross_val_score(pipe, X, y, cv=skf)
            all_scores.extend(scores)

        all_scores = np.array(all_scores)
        rows.append({
            "Classifier": name,
            "Label_level": label_level,
            "use_PCA": use_pca,
            "n_PCs": n_pcs if use_pca else None,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "Accuracy_mean": float(all_scores.mean()),
            "Accuracy_std": float(all_scores.std()),
        })

    results_df = pd.DataFrame(rows)
    return results_df


# ================================================================
# Downsampling RF accuracy (random loci / STRs)
# ================================================================

def downsampling_rf_accuracy(
    X,
    sample_df,
    test_nums=None,
    times=10,
    n_splits=5,
    random_state=42,
):
    """
    Downsampling test: randomly (with replacement) sample subsets of loci
    and evaluate RF accuracy for Superpopulation and Population labels.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix (samples x loci). This is geno_matrix.T.

    sample_df : pandas.DataFrame
        Metadata with index matching X.index, containing:
          - 'Superpopulation'
          - 'Population'

    test_nums : list of int or None
        Numbers of loci to sample each time.
        If None, defaults to [50, 100, 500, 1000, 2000, 3000, 4000, 5000].

    times : int
        Number of random repeats per feature count.

    n_splits : int
        Number of CV folds (StratifiedKFold).

    random_state : int
        Base random seed.

    Returns
    -------
    test_acc : pandas.DataFrame
        Columns:
          - 'num'   : number of loci used
          - 'level' : 'Superpopulation' or 'Population'
          - 'fold'  : CV fold index within each repeat
          - 'repeat': repeat index
          - 'accuracy' : accuracy for that fold
    """

    if test_nums is None:
        test_nums = [50, 100, 500, 1000, 2000, 3000, 4000, 5000]

    # Align labels
    common_samples = X.index.intersection(sample_df.index)
    X = X.loc[common_samples]
    y_super = sample_df.loc[common_samples, "Superpopulation"]
    y_pop = sample_df.loc[common_samples, "Population"]

    n_features_total = X.shape[1]
    print(f"Downsampling from total {n_features_total} loci.")

    # RF pipeline
    pipeline = Pipeline([
        ("imp", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=50,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    rng = np.random.default_rng(random_state)
    records = []

    for num in test_nums:
        if num <= 0:
            continue

        print(f"\nTesting num_features = {num}")
        for rep in range(times):
            # bootstrap loci (with replacement)
            cols_idx = rng.choice(X.shape[1], size=num, replace=True)
            subset_data = X.iloc[:, cols_idx]

            # Superpopulation
            skf_super = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state + rep,
            )
            fold_idx = 0
            for train_idx, test_idx in skf_super.split(subset_data, y_super):
                X_train, X_test = subset_data.iloc[train_idx], subset_data.iloc[test_idx]
                y_train, y_test = y_super.iloc[train_idx], y_super.iloc[test_idx]

                pipeline.fit(X_train, y_train)
                acc = pipeline.score(X_test, y_test)

                records.append({
                    "num": num,
                    "level": "Superpopulation",
                    "repeat": rep,
                    "fold": fold_idx,
                    "accuracy": float(acc),
                })
                fold_idx += 1

            # Population
            skf_pop = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state + rep,
            )
            fold_idx = 0
            for train_idx, test_idx in skf_pop.split(subset_data, y_pop):
                X_train, X_test = subset_data.iloc[train_idx], subset_data.iloc[test_idx]
                y_train, y_test = y_pop.iloc[train_idx], y_pop.iloc[test_idx]

                pipeline.fit(X_train, y_train)
                acc = pipeline.score(X_test, y_test)

                records.append({
                    "num": num,
                    "level": "Population",
                    "repeat": rep,
                    "fold": fold_idx,
                    "accuracy": float(acc),
                })
                fold_idx += 1

    test_acc = pd.DataFrame(records)
    return test_acc


# ================================================================
# Per-chromosome RF accuracy
# ================================================================

def per_chromosome_rf_accuracy(
    X,
    chr_list,
    sample_df,
    cv=5,
    scoring="accuracy",
    random_state=42,
):
    """
    Evaluate RF accuracy per chromosome.

    Parameters
    ----------
    X : pandas.DataFrame
        Genotype matrix (samples x loci).

    chr_list : array-like of str
        Chromosome label for each locus/column in X, e.g. ['chr1','chr1',...].

    sample_df : pandas.DataFrame
        Sample metadata with index matching X.index, containing:
          - 'Population'
          - 'Superpopulation'

    cv : int
        Number of CV folds.

    scoring : str
        Scoring metric for cross_val_score (default: 'accuracy').

    random_state : int
        Random seed for RF.

    Returns
    -------
    chr_acc : pandas.DataFrame
        Columns: ['chr', 'super', 'pop'], rows = per fold per chromosome.
    """

    # Align sample metadata
    common_samples = X.index.intersection(sample_df.index)
    X = X.loc[common_samples]
    meta = sample_df.loc[common_samples]

    y_pop = meta["Population"]
    y_super = meta["Superpopulation"]

    chr_list = np.asarray(chr_list)
    chr_acc = pd.DataFrame(columns=["chr", "super", "pop"])

    # RF pipeline
    pipeline = Pipeline([
        ("imp", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=50,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    for i in range(1, 23):
        cchr = f"chr{i}"
        print(cchr)

        chrom_idx = np.where(chr_list == cchr)[0]
        if chrom_idx.size == 0:
            print(f"  [INFO] No loci for {cchr}, skipping.")
            continue

        X_data = X.iloc[:, chrom_idx]

        cv_super = cross_val_score(
            pipeline, X_data, y_super, cv=cv, scoring=scoring
        )
        cv_pop = cross_val_score(
            pipeline, X_data, y_pop, cv=cv, scoring=scoring
        )

        chr_acc = pd.concat(
            [
                chr_acc,
                pd.DataFrame(
                    {
                        "chr": [cchr] * len(cv_super),
                        "super": cv_super,
                        "pop": cv_pop,
                    }
                ),
            ],
            ignore_index=True,
        )

    return chr_acc


# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Supervised population assignment from SNP/STR genotype matrices."
    )
    p.add_argument(
        "--matrix",
        required=True,
        help="Path to raw genotype matrix TSV (loci x samples).",
    )
    p.add_argument(
        "--sample-metadata",
        required=True,
        help="Path to sample metadata TSV with 'Superpopulation' and 'Population'.",
    )
    p.add_argument(
        "--label-level",
        choices=["Superpopulation", "Population"],
        default="Superpopulation",
        help="Target label level for main supervised task.",
    )
    p.add_argument(
        "--use-pca",
        action="store_true",
        help="Include PCA as feature reduction step before classifier.",
    )
    p.add_argument(
        "--n-pcs",
        type=int,
        default=50,
        help="Number of PCs to retain if --use-pca is set (default: 50).",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds for StratifiedKFold (default: 5).",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="How many times to repeat CV with different seeds (default: 1).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    p.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output TSV (e.g. results/1kgp_STR or results/hgdp_SNP).",
    )

    # Downsampling options
    p.add_argument(
        "--run-downsampling",
        action="store_true",
        help="Run downsampling RF test over random subsets of loci.",
    )
    p.add_argument(
        "--downsample-nums",
        type=str,
        default="50,100,500,1000,2000,3000,4000,5000",
        help="Comma-separated list of numbers of loci for downsampling (default: '50,100,500,1000,2000,3000,4000,5000').",
    )
    p.add_argument(
        "--downsample-times",
        type=int,
        default=10,
        help="Number of repeats per feature count in downsampling (default: 10).",
    )

    # Per-chromosome options
    p.add_argument(
        "--run-per-chrom",
        action="store_true",
        help="Run per-chromosome RF accuracy analysis.",
    )
    p.add_argument(
        "--chr-map",
        type=str,
        help="TSV with columns 'locus' and 'chr'; 'locus' must match genotype matrix row index.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading genotype matrix from: {args.matrix}")
    geno_df = pd.read_csv(args.matrix, sep="\t", index_col=0)

    print(f"Loading sample metadata from: {args.sample_metadata}")
    sample_df = pd.read_csv(args.sample_metadata, sep="\t", index_col=0)

    if args.label_level not in sample_df.columns:
        raise ValueError(f"Column '{args.label_level}' not found in sample metadata.")
    if "Superpopulation" not in sample_df.columns or "Population" not in sample_df.columns:
        raise ValueError("sample_metadata must contain 'Superpopulation' and 'Population' columns.")

    # Transpose loci x samples -> samples x loci
    X = geno_df.T
    print(f"Genotype matrix: {geno_df.shape[0]} loci x {geno_df.shape[1]} samples")
    print(f"Feature matrix for ML: {X.shape[0]} samples x {X.shape[1]} features")

    # 1) Main supervised assignment
    results_df = evaluate_supervised_assignment(
        X=X,
        sample_df=sample_df,
        label_level=args.label_level,
        use_pca=args.use_pca,
        n_pcs=args.n_pcs,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )

    out_main = f"{args.out_prefix}_supervised_assignment.tsv"
    print(f"Saving main supervised assignment results to: {out_main}")
    results_df.to_csv(out_main, sep="\t", index=False)

    # 2) Optional downsampling test
    if args.run_downsampling:
        test_nums = [int(x) for x in args.downsample_nums.split(",") if x.strip()]
        print(f"Running downsampling test with nums={test_nums}, times={args.downsample_times}")
        down_df = downsampling_rf_accuracy(
            X=X,
            sample_df=sample_df,
            test_nums=test_nums,
            times=args.downsample_times,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        out_down = f"{args.out_prefix}_downsampling_rf.tsv"
        print(f"Saving downsampling RF results to: {out_down}")
        down_df.to_csv(out_down, sep="\t", index=False)

    # 3) Optional per-chromosome accuracy
    if args.run_per_chrom:
        if args.chr_map is None:
            raise ValueError("--run-per-chrom requires --chr-map.")

        print(f"Loading chromosome map from: {args.chr_map}")
        chr_map = pd.read_csv(args.chr_map, sep="\t")

        if "locus" not in chr_map.columns or "chr" not in chr_map.columns:
            raise ValueError("--chr-map TSV must contain columns 'locus' and 'chr'.")

        # Align chromosomes to genotype matrix rows (loci)
        chr_map = chr_map.set_index("locus")
        missing = geno_df.index.difference(chr_map.index)
        if len(missing) > 0:
            print(f"[WARN] {len(missing)} loci in genotype matrix not found in chr-map; they will be ignored for per-chrom analysis.")

        # chr_list aligned with loci order in geno_df
        chr_list = chr_map.loc[geno_df.index, "chr"].values

        print("Running per-chromosome RF accuracy analysis ...")
        chr_acc_df = per_chromosome_rf_accuracy(
            X=X,
            chr_list=chr_list,
            sample_df=sample_df,
            cv=args.n_splits,
            scoring="accuracy",
            random_state=args.random_state,
        )

        out_chr = f"{args.out_prefix}_perchr_rf.tsv"
        print(f"Saving per-chromosome RF results to: {out_chr}")
        chr_acc_df.to_csv(out_chr, sep="\t", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
