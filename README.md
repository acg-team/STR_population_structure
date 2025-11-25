# Population Structure Inference using Genome-wide STR variations

This repository contains a Python pipeline for analyzing genome-wide Short Tandem Repeat (STR) variation in human populations. The pipeline was applied to the following datasets:
+ 1000 Genomes Project (1KGP)
+ Human Genome Diversity Project (HGDP)
+ Simon Genome Diversity Project (SGDP)
+ H3Africa

<p align="center">
  <img src="figures/overview.jpg" width="700">
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
links to be a


### Sample metadata

Columns should include:

- sample ID (used to align with matrix columns)
- `Superpopulation` (continental population labels)
- `Population` (regional population labels)
- optional extra annotations

---

## 2. Key Components of the Pipeline

It includes:

- STR matrix construction and filtering
- Cross-dataset STR overlap and batch-effect filtering
- Population structure analysis (PCA, unsupervised clustering)
- Supervised population assignment (Random Forest, Naive Bayes)
- Directional NMF for STR-based admixture inference
- Characterization of ancestry-informative STR signatures2. STR Matrices and Overlap Between Datasets

### STR Matrix Construction
Scripts convert EnsembleTR/HipSTR calls into per-chromosome or genome-wide matrices, apply variance filtering, and generate `str_info.tsv`.

### Cross-dataset STR Harmonization
- Genomic overlap using PyRanges  
- Variance filtering  
- Removal of batch-specific loci by comparing population-level STR means  

Produces matched STR matrices for multi-dataset population analyses.

### Population Structure analysis
Implemented in Python:
- PCA  
- K-means clustering  
- Adjusted Rand Index (ARI) evaluation  
- Mantel tests comparing STR/SNP distances to geographic distances  

### Supervised Population Assignment

Models:
- Random Forest  
- Naive Bayes  
With:
- optional PCA  
- cross-validation  
- downsampling experiments  
- per-chromosome accuracy evaluation  

### Directional NMF STR-based admixture inference

Directional NMF is applied to:
- positive (expansion) matrix  
- negative (contraction) matrix  

Components are matched using the Hungarian algorithm.  
For each component:
- top expansion loci and top contraction loci are extracted  
- driver set = union of expansion + contraction loci  

Enrichment analysis (motif, period) uses Fisher’s exact test with FDR correction.

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