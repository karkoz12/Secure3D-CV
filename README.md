Secure3D-CV — Reference Implementation
Short Introduction

This repository provides a reference Python implementation of Secure3D-CV, a hybrid method for detecting abnormal and adversarial behavior in 3D computer vision systems. The code accompanies the manuscript and is intended to support reproducibility, transparency, and methodological clarity, rather than serving as a production-ready system.

Secure3D-CV combines statistical anomaly detection with a learned classifier to synthesize a unified risk score for 3D scenes represented as point clouds. The repository includes tools for synthetic data generation, attack simulation, feature extraction, model training, and evaluation, closely following the structure described in the paper.

Repository Structure
.
├── secure3d_cv_data_generator.py   # Synthetic 3D data and attack simulation
├── secure3d_cv_system.py           # Secure3D-CV core logic and evaluation
├── README.md
├── LICENSE

Overview of the Method

Secure3D-CV operates in three main stages:

3D Scene Simulation and Feature Extraction
Synthetic 3D point clouds are generated to represent benign scenes. Two attack types are simulated:

perturbation (noise injection),

point cloud tampering (point injection).

From each scene, compact statistical and geometric features are extracted.

Hybrid Risk Modeling

A statistical anomaly detector (Isolation Forest) models normal behavior.

A learned classifier (logistic regression) distinguishes benign from attacked samples.

Both outputs are combined into a hybrid risk score using a weighting parameter α.

Decision and Evaluation
Final predictions are derived by thresholding the hybrid risk score. Performance is evaluated using standard metrics such as AUC, F1-score, TPR, and FPR, as reported in the manuscript.

Requirements

Python ≥ 3.9

NumPy

Pandas

scikit-learn

Install dependencies with:

pip install numpy pandas scikit-learn

Usage
1. Generate a Synthetic Dataset
python secure3d_cv_data_generator.py


This generates benign and attacked 3D scenes, extracts features, and prints basic statistics.

2. Train and Evaluate Secure3D-CV
python secure3d_cv_system.py


This will:

generate a dataset,

train the Secure3D-CV model,

evaluate performance at the optimal α value,

perform a sensitivity analysis over α.

Notes on Reproducibility

The data generation process is synthetic and simplified by design.

Numerical results will not exactly match the paper’s reported values, but qualitative trends (e.g., optimal α region, robustness gain over baseline) are preserved.

The code is intended for methodological illustration and reproducibility, not for deployment in safety-critical systems.

License

This project is released under the Apache License 2.0.
See the LICENSE file for details.
