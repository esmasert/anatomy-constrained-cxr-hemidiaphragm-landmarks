# anatomy-constrained-cxr-hemidiaphragm-landmarks
Anatomy-constrained chest X-ray pipeline: lung segmentation + two-stage heatmap regression for hemidiaphragm landmark localisation.

Anatomy-Constrained Hemidiaphragm Landmark Localisation on Chest X-Ray Images

An open baseline for hemidiaphragm landmark localisation on chest X-rays using an anatomy-constrained pipeline (lung segmentation → two-stage heatmap regression → evaluation). This repository accompanies the manuscript “Accessible Hemidiaphragm Landmark Localisation on Chest X-Ray Images Using Deep Learning and Anatomical Constraints.”

Overview

Anatomy-aware approach combining lung segmentation with heatmap-based landmark localisation.

Reproducible preprocessing, training, testing, and evaluation stages.

Modular structure to plug in different backbones or training regimes.

Pipeline

Lung Segmentation
Train a segmentation model to obtain lung masks used as anatomical priors.

Heatmap Data Preprocessing
Prepare final CSV/metadata from masks and reference points for the heatmap stage.

Heatmap Regression
Train a two-stage heatmap model to localise hemidiaphragm landmarks.

Evaluation & Testing
Report segmentation Dice/IoU and landmark error metrics (mean/median pixel error, PCK at fixed thresholds) with qualitative overlays.

Repository Contents

Segmentation – Training
Scripts to train a lung segmentation model.

Segmentation – Testing
Utilities to evaluate segmentation checkpoints and summarise metrics.

Heatmap – Data Preprocessing
Tools to reshape/merge CSVs and generate the final metadata for landmark training.

Heatmap – Training
Training entry points for the two-stage heatmap regression.

Heatmap – Testing
Inference/evaluation utilities for predicted landmarks and heatmaps.

Notebooks (optional)
Sanity checks, quick visualisations, and qualitative examples.

Data

This repository does not include datasets. Please obtain data from the original sources and follow their licenses. Prepare dataset directories and split files as required by your environment. Example folder conventions and expected CSV fields are described in the preprocessing scripts.

Results

Add your validation/test results here, including:

Segmentation: Dice / IoU on lung masks.

Landmarks: mean and median pixel error, PCK at chosen thresholds, and qualitative overlays.

How to Cite

Sert, E., Azimbagirad, M., Onah, D., Alexander, D. C., Jacob, J., & Aslani, S.
Accessible Hemidiaphragm Landmark Localisation on Chest X-Ray Images Using Deep Learning and Anatomical Constraints.
Submitted to IEEE ISBI 2026, 2025.

A machine-readable citation file is provided as CITATION.cff. Update it (and this section) with arXiv/DOI details when available.

License

Code is released under Apache-2.0.
Datasets are not distributed; please respect original data licenses.

Acknowledgements

We thank our collaborators and institutions listed in the manuscript. Ethics/compliance and funding acknowledgements are detailed in the paper.
