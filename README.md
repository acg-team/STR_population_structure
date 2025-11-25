# Population Structure Inference using Genome-wide STR variations

This repository contains a Python pipeline for analyzing genome-wide Short Tandem Repeat (STR) variation in human populations. It includes:

- STR matrix construction and filtering
- Cross-dataset STR overlap and batch-effect filtering
- Population structure analysis (PCA, unsupervised clustering)
- Supervised population assignment (Random Forest, Naive Bayes)
- Directional NMF for STR-based admixture inference
- Characterization of ancestry-informative STR signatures

The code can be applied to the following datasets:
+ 1000 Genomes Project (1KGP)
+ Human Genome Diversity Project (HGDP)
+ Simon Genome Diversity Project (SGDP)
+ H3Africa

<p align="center">
  <img src="figures/overview.jpg" width="700">
</p>

## 1. Input Formats

Before running the pipeline, ensure the following input files are available.

### 1.1 STR Genotyping VCFs

The pipeline uses STR genotypes of 1KGP and H3Africa from:

- **EnsembleTR** (recommended):  
  GitHub: https://github.com/gymreklab/EnsembleTR  
  (Supports multi-caller consensus STR genotyping; HipSTR output is available via `METHODS[2] == 1`.)
  
STR genotypes of HGDP and SGDP using HipSTR:

- **HipSTR**:  
  https://github.com/tfwillems/HipSTR

### 1.2 Sample metadata

Columns should include:

- sample ID (used to align with matrix columns)
- `Superpopulation` (continental population labels)
- `Population` (regional population labels)
- optional extra annotations

---

## 2. STR Matrices and Overlap Between Datasets

These scripts construct STR genotype matrices and align loci between datasets.

### STR matrices

Input:

- STR genotype matrix from EnsembleTR or HipSTR (VCF-based)
- STR info file with columns such as:
  - `chr`, `start`, `end`
  - `str_id`
  - `period`
  - `str_var`
  - `methods` (for EnsembleTR, used to select HipSTR output)

Processing:

- Filter STRs by variance (`str_var` threshold)
- Create per-chromosome and genome-wide STR matrices:
  - rows = STRs (index = `str_uid`)
  - columns = samples
  - values = mean STR length or copy number

### Overlap and batch-effect filtering

Script: e.g. `overlap_batch_filter_strs.py`

Steps:

1. Read two STR info files (dataset 1 and dataset 2)
2. Filter STRs by variance in each dataset
3. Use genomic coordinates to find overlapping loci between datasets (PyRanges)
4. Subset genotype matrices to the overlapping STRs
5. Remove loci with strong batch effects using population-specific mean differences:

   - For each population (e.g. AFR, EAS, EUR):
     - Compute mean STR length in dataset 1 and dataset 2
     - Keep loci where absolute mean difference is below a threshold (e.g. 1 repeat unit) in all selected populations

Output:

- Cleaned STR info for dataset 1 and dataset 2 (matched loci)
- Cleaned STR matrices for dataset 1 and dataset 2 (same loci, aligned)
- Summary file: number of loci before/after batch-effect filtering

## 3. Population Structure Analysis

### PCA and K-means clustering

Scripts take as input:

- STR or SNP genotype matrix (samples x loci)
- Sample metadata with:
  - individual ID
  - superpopulation label
  - population label

Analyses:

- Perform PCA on the genotype matrix
- For a range of numbers of PCs (e.g. 1 to 50):
  - Run K-means clustering with multiple random seeds
  - Evaluate clustering with Adjusted Rand Index (ARI) at:
    - superpopulation level
    - population level within each superpopulation

Output:

- Per-configuration ARI values
- Aggregated mean and standard deviation of ARI across repeats

### Mantel test with geography

Script: e.g. `14_geo_mantel_test.py`

Inputs:

- STR population distance matrix (e.g. Goldstein δμ² between populations)
- SNP population distance matrix (allele sharing)
- Geographic coordinates per population (latitude, longitude)
- Population labels and superpopulation groupings

Steps:

- Compute geographic distance matrix (Haversine)
- For each superpopulation:
  - Restrict to populations in that group
  - Perform Mantel tests:
    - STR distance vs geographic distance
    - SNP distance vs geographic distance

Output:

- Mantel correlation coefficients (r) and p-values per superpopulation and marker type (STR, SNP)

## 4. Supervised Population Assignment

Script: e.g. `16_supervised_assignment.py`

Inputs:

- Genotype matrix (loci x samples) for STRs or SNPs
- Sample metadata with:
  - sample ID as index or column
  - `Superpopulation`
  - `Population`

Main features:

- Supervised classification using:
  - Random Forest (RF)
  - Gaussian Naive Bayes (NB)
- Target labels:
  - Superpopulation
  - Population
- Optional PCA-based dimensionality reduction
- Stratified K-fold cross-validation (default 5-fold)
- Repeat CV with multiple seeds

Outputs:

- Classification accuracy (mean and standard deviation) for each classifier and label level
- Optional:
  - Downsampling tests: accuracy as a function of number of loci
  - Per-chromosome accuracy: RF performance using loci from each chromosome separately

Example usage:

- Superpopulation classification with PCA:
  - `--label-level Superpopulation --use-pca`

- Population classification without PCA:
  - `--label-level Population`

## 5. Directional-NMF (dNMF) admixture inference

Script: e.g. `nmf_cv.py`

Inputs:

- STR genotype matrix (rows = STRs, columns = samples)
- Optional STR info file (for filtering on period, variance, and HipSTR calls)

Processing:

1. Load STR matrix
2. Optionally subset STRs based on:
   - variance (`str_var` > threshold)
   - period (e.g. period = 2, 3, 4)
   - method (HipSTR subset from EnsembleTR)
3. Transform and standardize:
   - transpose to samples x STRs
   - impute missing values by feature mean
   - standardize features (mean 0, variance 1)
4. Split into positive and negative matrices:
   - `X_pos = max(X, 0)` (expansions above mean)
   - `X_neg = max(-X, 0)` (contractions below mean)
5. For K in a given range (e.g. 3 to max_k):
   - Fit NMF on `X_pos` and `X_neg` separately
   - Obtain:
     - Q_pos, H_pos
     - Q_neg, H_neg
   - Match components between Q_pos and Q_neg using the Hungarian algorithm
   - Compute:
     - correlation of matched components
     - reconstruction errors

Usage with Slurm array:

- Each Slurm task runs one repeat with a different random seed
- Results for different repeats are combined downstream

Outputs:

- Per-K metrics (e.g. mean correlation, variance of correlation, reconstruction error) saved as CSV per repeat

## 6. Ancestry-informative STR Signatures Extraction

### Component matching (Q-space)

Given saved matrices:

- `Q_pos` (samples x K)
- `Q_neg` (samples x K)

Steps:

1. Compute correlation matrix between Q_pos and Q_neg components
2. Use the Hungarian algorithm (linear sum assignment) on the negative absolute correlation as cost
3. Match each positive component to a negative component
4. Store:
   - matching indices
   - per-component correlations

### Expansion and contraction signatures (H-space)

Given saved matrices:

- `H_pos` and `H_neg` (saved in some convention; transposed to STRs x K)

Steps:

1. Reorder columns of H_pos and H_neg using the matched indices from Q-space
2. For each component:
   - compute Pearson correlation between the column in H_pos and the corresponding column in H_neg
3. For each component k:
   - select top X% of STRs by loading in H_pos (expansion-associated loci)
   - select top X% of STRs by loading in H_neg (contraction-associated loci)
   - form the union set of STR indices for that component

Important:

- Expansion and contraction signatures are kept separate.
- There is no subtraction H_pos - H_neg.
- For each NMF component:
  - expansion signature: top loci from H_pos
  - contraction signature: top loci from H_neg
  - combined driver set: union of top expansion and contraction loci

This design reflects the bidirectional nature of STR mutation, allowing expansions and contractions to have different patterns while still defining a single biological “process” per component.

## 7. Enrichment Testing for STR Signatures

Script: e.g. `extract_signatures_enrichment.py`

Inputs:

- `str_info` table with:
  - `str_uid` (matching the STR matrix index)
  - one or more annotation columns for testing, e.g.:
    - `period`
    - `motif`
- STR data matrix used in NMF, with index = `str_uid`
- dictionaries of component-specific STR sets (e.g. union of top expansion and contraction loci)
- target annotation column name (e.g. `period`)

Method:

For each component:

1. Take the set of driver loci (indices into the STR matrix)
2. Map those indices to `str_uid` and then to `str_info`
3. Count how many STRs in the driver set have each value of the target annotation (e.g. each period)
4. Build a 2x2 contingency table for each annotation category:
   - in driver set vs not in driver set
   - with given category vs all other categories
5. Perform Fisher's exact test (one-sided, enrichment)
6. Compute enrichment ratio:
   - (proportion of category in driver set) / (proportion of category in background)
7. Apply Benjamini–Hochberg FDR correction across all tests

Output:

- A table with one row per component and annotation category:
  - component ID
  - annotation value (e.g. period)
  - enrichment ratio
  - odds ratio
  - raw p-value
  - FDR-adjusted p-value
  - significance flag


## 8. Dependencies

- Python 3.8 or newer
- numpy
- pandas
- scipy
- scikit-learn
- statsmodels
- pyranges

Installable via pip or conda, for example:

```bash
pip install numpy pandas scipy scikit-learn statsmodels pyranges

