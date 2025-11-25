# Population Structure Inference using Genome-wide STR variations

This repository contains a Python pipeline for analyzing genome-wide Short Tandem Repeat (STR) variations in human populations. The pipeline was applied to the following datasets:
+ 1000 Genomes Project (1KGP)
+ Human Genome Diversity Project (HGDP)
+ Simon Genome Diversity Project (SGDP)
+ H3Africa

<p align="center">
  <img src="figures/overview.jpg">
</p>


## 1. Input Formats

Before running the pipeline, ensure the following input files are available.

###  STR Genotyping VCFs

The pipeline uses STR genotypes of 1KGP and H3Africa from:

- **EnsembleTR** (recommended):  
  GitHub: https://github.com/gymreklab/EnsembleTR  
  (Supports multi-caller consensus STR genotyping; HipSTR output is available via `METHODS[2] == 1`.)
  
STR genotypes of HGDP and SGDP using HipSTR:

- **HipSTR**:  
links to be added


### Sample metadata

Columns should include:

- Sample ID (used to align with matrix columns)
- Continental population labels
- Regional population labels

---

## 2. Key Components of the Pipeline

`scripts/` includes:

- STR matrix construction and filtering
- Cross-dataset STR overlap and batch-effect filtering
- Population structure analysis (PCA, unsupervised clustering)
- Supervised population assignment (Random Forest, Naive Bayes)
- Directional NMF for STR-based admixture inference
- Characterization of ancestry-informative STR signatures2. STR Matrices and Overlap Between Datasets

## STR Matrices and Overlap Between Datasets

### STR Matrix Construction

Scripts convert EnsembleTR or HipSTR VCFs into per-chromosome and genome-wide STR matrices.

Main steps:
- parse STR genotypes (`NCOPY`, `GB`)
- filter loci by motif period and variance
- optional: keep only HipSTR calls from EnsembleTR (`METHODS[2] == 1`)
- produce:
  - `str_info.tsv` (metadata for STR loci)
  - `str_matrix.tsv` (samples × STR loci)

### Cross-dataset STR Harmonization

Pipeline steps:
- find genomic overlaps using PyRanges
- filter loci by variance within each dataset
- remove batch-driven loci by comparing population-level STR means across datasets
- generate merged STR matrices aligned to the same set of loci

These matrices are used for joint population analyses.

---

## Population Structure Analysis (STR vs SNP)

All analyses are implemented in Python.

Features:
- PCA-based dimensionality reduction
- K-means clustering 
- Adjusted Rand Index (ARI) evaluation at:
  - continental population levels
  - regional population levels
- hierarchical clustering of genetic distance matrices (STRs and SNPs)
- comparison of STR/SNP distances to geographic distances using Mantel tests

This module compares the resolution of SNPs vs STRs for global population inference.

---

## Supervised Population Assignment (STR vs SNP)

Two classifiers are implemented:
- Random Forest
- Naive Bayes

Features:
- optional PCA for feature reduction
- 5-fold cross-validation
- comparison of SNP-based vs STR-based accuracy
- downsampling tests (accuracy vs number of STR loci)
- per-chromosome accuracy evaluation

Outputs include classification accuracy at both superpopulation and population levels.

---

## Directional NMF for STR-based Admixture Inference

Directional NMF (dNMF) decomposes standardized STR data into two parts:
- expansion-associated variation
- contraction-associated variation

The model assumes ancestral components are encoded in both bidirectional STR mutations, expansion and contraction.

Input: STR genotype matrix from 1KGP and HGDP+SGDP datasets
Output: Ancestry coefficients for each individual and ancestry-informative STR signatures

We applied dNMF to the 1KGP and HGDP+SGDP, 12 and 11 ancestral populations were detect in each dataset.

<p align="center">
  <img src="figures/res_dnmf.jpg">
</p>

---

## Ancestry-informative STR Signatures

Mutation signatures are defined by the union of loci with:
- high expansion loadings (from `H_pos`)
- high contraction loadings (from `H_neg`)

Signature enrichment analysis:
- motif and period enrichment tests
- Fisher’s exact test
- FDR correction (Benjamini–Hochberg)

These signatures highlight STR features enriched in specific ancestral components.


## 3. Notebooks

This repository includes Jupyter notebooks for visualizations in the `notebooks/` and `R_scripts/` directory.

## 4. Dependencies

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
```

## 5. Citation