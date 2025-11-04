# Anatomy-Constrained CXR Hemidiaphragm Landmarks

> Anatomy-constrained chest X-ray pipeline: lung segmentation + two-stage heatmap regression for hemidiaphragm landmark localisation.

An open baseline for hemidiaphragm landmark localisation on chest X-rays using an anatomy-constrained pipeline (lung segmentation → two-stage heatmap regression → evaluation). This repository accompanies the manuscript **"Accessible Hemidiaphragm Landmark Localisation on Chest X-Ray Images Using Deep Learning and Anatomical Constraints."**

## Overview

- **Anatomy-aware approach** combining lung segmentation with heatmap-based landmark localisation
- **Reproducible pipeline** with preprocessing, training, testing, and evaluation stages
- **Modular structure** to plug in different backbones or training regimes

## Pipeline

### 1. Lung Segmentation
Train a segmentation model to obtain lung masks used as anatomical priors.

### 2. Heatmap Data Preprocessing
Prepare final CSV/metadata from masks and reference points for the heatmap stage.

### 3. Heatmap Regression
Train a two-stage heatmap model to localise hemidiaphragm landmarks.

### 4. Evaluation & Testing
Report segmentation Dice/IoU and landmark error metrics (mean/median pixel error, PCK at fixed thresholds) with qualitative overlays.

## Repository Contents

### 📁 Segmentation
- **Training**: Scripts to train a lung segmentation model
- **Testing**: Utilities to evaluate segmentation checkpoints and summarise metrics

### 📁 Heatmap
- **Data Preprocessing**: Tools to reshape/merge CSVs and generate the final metadata for landmark training
- **Training**: Training entry points for the two-stage heatmap regression
- **Testing**: Inference/evaluation utilities for predicted landmarks and heatmaps

### 📁 Notebooks (Optional)
- Sanity checks, quick visualisations, and qualitative examples

## 📊 Data

⚠️ **Important**: This repository does not include datasets. 

- Please obtain data from the original sources and follow their licenses
- Prepare dataset directories and split files as required by your environment
- Example folder conventions and expected CSV fields are described in the preprocessing scripts

## 📈 Results

Add your validation/test results here, including:

### Segmentation Metrics
- **Dice coefficient** and **IoU** on lung masks

### Landmark Localisation Metrics
- Mean and median pixel error
- **PCK** (Percentage of Correct Keypoints) at chosen thresholds
- Qualitative overlays and visualisations

## 📝 How to Cite

If you use this code in your research, please cite:

```bibtex
@article{sert2025accessible,
  title={Accessible Hemidiaphragm Landmark Localisation on Chest X-Ray Images Using Deep Learning and Anatomical Constraints},
  author={Sert, E. and Azimbagirad, M. and Onah, D. and Alexander, D. C. and Jacob, J. and Aslani, S.},
  journal={Submitted to IEEE ISBI 2026},
  year={2025}
}
```

📄 A machine-readable citation file is provided as `CITATION.cff`. Update it (and this section) with arXiv/DOI details when available.

## 📄 License

- **Code**: Released under [Apache-2.0](LICENSE)
- **Datasets**: Not distributed; please respect original data licenses

## 🙏 Acknowledgements

We thank our collaborators and institutions listed in the manuscript. Ethics/compliance and funding acknowledgements are detailed in the paper.

---

⭐ **Star this repository** if you find it useful for your research!
