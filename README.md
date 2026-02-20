# Code and Software for: Prefrontal scaling of reward prediction error readout 1 gates reinforcement-derived adaptive behavior in primates

## Description

This repository contains the source code, processed data, and analysis pipelines required to reproduce the findings of our study. The project integrates behavioral modeling (Python), bioinformatics/gene analysis (R), and single-trial prediction models (Python). For the fMRI univariate analysis (MATLAB/SPM), specific regressor settings and GLM parameters are detailed in the Methods section of the manuscript.

---

## 1. System Requirements

### Software Dependencies

The analysis is distributed across three main environments:

**A. Behavioral & Cognitive Modeling (Python)**

- **Python Version:** 3.10.14
- **Core Packages:** `pymc` (v5.15.1), `arviz` (v0.18.0), `pytensor` (v2.13.1), `numpy` (v1.26.4), `zarr` (v2.13.3).

**B. Univariate GLM Analysis (MATLAB)**

- **MATLAB Version:** R2020a
- **Toolbox:** SPM12 (v7771)

**C. Bioinformatics & Gene Analysis (R)**

- **R Version:** (e.g., 4.2.0)
- **Packages:** `Limma` (v3.52.4), `ggplot2` (v3.4.2), `clusterProfiler` (v4.7.1), `org.Mmu.eg.db` (v3.15.0), `Seurat` (v4.3.0), `AUCell` (v1.16.0), `org.Hs.eg.db` (v3.15.0), `homologene` (v1.1.68).

**D. Single-Trial Prediction & SVM (Python)**

- **Python Version:** 3.13.1
- **Core Packages:** `scikit-learn` (v1.6.0), `scipy` (v1.14.1), `numpy` (v2.1.3).

### Hardware Requirements

- **Standard Desktop Computer:** All analyses can be run on a standard computer with at least 16GB RAM. No non-standard hardware is required.

---

## 2. Installation Guide

We recommend using `conda` to manage the different Python environments to ensure version compatibility.

### Step 1: Environment for Cognitive Modeling (Python 3.10)

Following [PyMC official recommendations](https://www.pymc.io/projects/docs/en/latest/installation.html), use the `conda-forge` channel:

```bash
conda create -c conda-forge -n bhv_model python=3.10.14 pymc=5.15.1 numpy=1.26.4 arviz=0.18.0 pytensor=2.13.1 zarr=2.13.3
```

### Step 2: Environment for Single-Trial Prediction (Python 3.13)

```bash
conda create -n prediction_model python=3.13.1 numpy=2.1.3 scipy=1.14.1 scikit-learn=1.6.0
```

### Step 3: R & MATLAB Setup

- **R:** Install the listed packages via `BiocManager::install()` or `install.packages()`.
- **MATLAB:** Add SPM12 to your MATLAB path.

**Typical Install Time:** ~15 minutes.

---

## 3. Demo & Instructions for Use

### Behavioral Analysis & Modeling Demo

- **File:** `cognitive_modeling.ipynb`
- **Data:** Uses sample data for 3 human participants in `/data/`.
- **Expected Output:** Parameter fits for alpha and beta, along with learning curve visualizations.
- **Expected Run Time:** < 30 seconds for basic analysis. For parameter fitting with 3 participants, the run time is minimal but scales with the number of subjects and free parameters.

### Single-Trial Prediction Demo

- **File:** `single_trial_prediciton.ipynb`
- **Data:** Uses sample data for 3 participants.
- **Expected Output:** Model framework verification and prediction accuracy scores.
- **Expected Run Time:** Based on the manuscript settings (100 repetitions of 5-fold cross-validation), the expected run time is approximately 1–2 hours.

### Instructions for use on your own data

To apply these analytical pipelines to your own data, please follow the specific guidelines for each module:

**A. Behavioral Analysis & Cognitive Modeling**

- **Data Loading:** Use the `read_csv_raw_data` function located in `preprocessing.py` to import your behavioral logs.
- **Normalization:** You must apply the mapping dictionaries (defined in `preprocessing.py`) to standardize `action` and `reward` columns. This ensures that the reinforcement learning models and aggregation scripts can correctly index the choice behavior and outcome variables.

**B. fMRI Univariate Analysis (SPM)**

- **GLM Configuration:** When setting up the 1st-level analysis in SPM12, ensure that the GLM conditions (regressors) are configured exactly as described in the "**Model-based fMRI Data Analysis"** subsection of the Methods. Pay particular attention to the onset timing and parametric modulators for events.

**C. Transcriptomics & Gene Analysis**

- **Data Acquisition:** Due to copyright and licensing restrictions, the raw human gene expression data (AHBA) is not included in this repository. Users must download the transcriptomic datasets from the original sources cited in the manuscript.
- **Preprocessing:** Before running the R scripts in `Bioinfomatics_Code_Archive`, ensure that all spatial gene expression maps are accurately registered to the **standard MNI space** to match the fMRI parcel boundaries.

**D. Single-trial Prediction Pipeline**

- **Step 1 (Neural Modeling):** First, perform single-trial activation estimation using the **Least Squares Separate (LSS)** approach.
- **Step 2 (Prediction):** Use the derived neural feature matrices to construct prediction models for the **next-trial** behavior or latent parameters. The framework for this nested cross-validation is demonstrated in `single_trial_prediciton.ipynb`.

---

## 4. Repository Structure

- `bhv_analysis.py`: computes accuracy, miss rate, reaction time, and aggregates performance around reversal points.
- `preprocessing.py`: loads CSV behavioral logs, handles label mappings/missing trials, pads variable-length runs, and gathers file lists.
- `self_define_general_name.py`: maps Glasser atlas parcel indices and area names, counts voxels in masks, and looks up human behavioral IDs.
- `visualization.py`: plotting helpers for violin/box plots, CR time courses, reversal performance, confusion matrices, correlations, permutation summaries, and single-run timelines.
- `Fig1.ipynb`, `Fig4.ipynb`, `FigS2.ipynb`, `FigS3.ipynb`: figure-generation notebooks for the manuscript.
- `Fig5.ipynb`: alignment/accuracy figure notebook (permutation and real prediction results).
- `FigS1.ipynb`: compares model fits (Delta ELPD WAIC, EMF, EP).
- `figS5.ipynb`: plots tSNR voxel-distribution histograms.
- `data/`: inputs for notebooks and scripts (behavioral CSVs, model outputs, and permutation results).
- `Bioinfomatics_Code_Archive/`: Contains R scripts and specific README for Fig3/FigS7.

---

## License

This project is licensed under the MIT License.