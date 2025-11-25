#!/usr/bin/env python3
"""
Supervised population assignment from raw SNP/STR genotype matrices.

Input:
  - Raw genotype matrix (loci x samples), SNP or STR
  - Sample metadata with columns:
        'Superpopulation', 'Population'
    and index = sample IDs matching the matrix columns.

Pipeline:
  - Align samples between matrix and metadata
  - Optionally apply PCA for feature reduction
  - Train + evaluate:
        - RandomForestClassifier (RF)
        - GaussianNB (NB)
    via stratified K-fold cross-validation

Output:
  - <out_prefix>_supervised_assignment.tsv
    (mean / std accuracy for each classifier)
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
        Column in sample_df used as the target (e.g. "Superpopulation", "Population").

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
    imputation = SimpleImputer(missing_values=np.nan, strategy="mean")
    scaler = StandardScaler()

    # Classifiers
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=50,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )

    nb = GaussianNB()

    def make_pipeline(clf):
        steps = [("imp", imputation), ("scaler", scaler)]
        if use_pca:
            steps.append(("pca", PCA(n_components=n_pcs, random_state=random_state)))
        steps.append(("clf", clf))
        return Pipeline(steps)

    pipelines = {
        "RF": make_pipeline(rf),
        "NB": make_pipeline(nb),
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
        help="Target label level: 'Superpopulation' or 'Population'.",
    )
    p.add_argument(
        "--use-pca",
        action="store_true",
        help="Include PCA as feature reduction step before classifier.",
    )
    p.add_argument(
        "--n-pcs",
        type=int,
        default=30,
        help="Number of PCs to retain if --use-pca is set (default: 30).",
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
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading genotype matrix from: {args.matrix}")
    geno_df = pd.read_csv(args.matrix, sep="\t", index_col=0)

    print(f"Loading sample metadata from: {args.sample_metadata}")
    sample_df = pd.read_csv(args.sample_metadata, sep="\t", index_col=0)

    if args.label_level not in sample_df.columns:
        raise ValueError(f"Column '{args.label_level}' not found in sample metadata.")

    # Transpose loci x samples -> samples x loci
    X = geno_df.T
    print(f"Genotype matrix: {geno_df.shape[0]} loci x {geno_df.shape[1]} samples")
    print(f"Feature matrix for ML: {X.shape[0]} samples x {X.shape[1]} features")

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

    out_path = f"{args.out_prefix}_supervised_assignment.tsv"
    print(f"Saving results to: {out_path}")
    results_df.to_csv(out_path, sep="\t", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
