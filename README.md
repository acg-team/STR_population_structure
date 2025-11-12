# STR-based Population Structure Inference

## Overview
This repository contains the implementation of multi-model framework for STR-based population structure, including a novel admixture inference model, Directional Non-negative Matrix Factorization (dNMF) and related analyses from the study:
> High-resolution population structure inference using genome-wide STR variations.

## Methods
Short tandem repeats (STRs) are a major yet underexplored source of human genetic variation. This framework integrates:

+ Unsupervised clustering (PCA, t-SNE)
+ Supervised population assignment
+ Directional NMF (dNMF), a novel admixture-inspired model that infers ancestry coefficients by explicitly modeling the bidirectional mutation dynamics of STRs.

## Datasets
+ 1000 Genomes Project (1KGP)
+ Human Genome Diversity Project (HGDP)
+ Simon Genome Diversity Project (SGDP)
+ H3Africa

