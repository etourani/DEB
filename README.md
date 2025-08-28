# DEB
README - Directional Entropy Bands (DEB) and SOAP Descriptors for Polymers
Supplementary Material for:
"Directional Entropy Bands for Surface Characterization of Polymer Crystallization"
E. Tourani, B. J. Edwards, B. Khomami, 2025

----------------------------------------------------------------------
Repository Structure
----------------------------------------------------------------------

SM_DEB_SOAP/
  classification/    Supervised ML scripts, datasets, and results
  DEB/               Directional Entropy Band (DEB) descriptor calculation
  SOAP/              SOAP descriptor calculation and related outputs

----------------------------------------------------------------------
1. Classification Module  (SM_DEB_SOAP/classification/)
----------------------------------------------------------------------

Contains labeled datasets, Python script for supervised classification,
and performance results.

- tmid_DEB_012labels.csv       Labeled dataset (surface vs melt/crystal)
- 1.py                         Classification pipeline (RF, LogReg, XGBoost)
- metrics_barplot.png/.pdf     Barplots of performance metrics
- Confusion matrices:
    confmat_LR_surface_melt.*      Logistic Regression (surface vs melt)
    confmat_LR_surface_crystal.*   Logistic Regression (surface vs crystal)
    confmat_RF.*                   Random Forest (surface vs rest)
    confmat_GB.*                   Gradient Boosting (surface vs rest)

----------------------------------------------------------------------
2. DEB Module  (SM_DEB_SOAP/DEB/)
----------------------------------------------------------------------

Scripts and outputs for calculating Directional Entropy Band (DEB) descriptors.

- deb.py                         Core DEB calculation pipeline
- tmid_Si_Savg.csv               Scalar and averaged band entropy values
- requirements.txt               Python dependencies
- deb_out/                       Output directory
    deb_features.csv              DEB descriptors per atom
    deb_umap2d.csv / .png         UMAP projections of DEB features
    config.json                   Configuration used
    run.log                       Execution log

----------------------------------------------------------------------
3. SOAP Module  (SM_DEB_SOAP/SOAP/)
----------------------------------------------------------------------

Scripts and outputs for SOAP descriptor computation.

- soap_pipeline.py                Main SOAP calculation pipeline
- combinedOPs_tmid_labeled.csv    Combined SOAP + order parameters (labeled)
- requirements.txt                Python dependencies
- soap_out/                       Output directory
    soap_features.npz              Raw SOAP feature arrays
    pca_variance.csv / .png        PCA explained variance
    umap2d.csv / labeled.csv       UMAP projections
    umap2d_hdbscan.png             UMAP + HDBSCAN clustering
    clustered_atoms_soap.xyz       Clustered atomic structure (XYZ format)
    config.json                    Configuration used
    run.log                        Execution log

----------------------------------------------------------------------
4. Requirements
----------------------------------------------------------------------

Both DEB and SOAP modules include requirements.txt files.
Install dependencies using:

    pip install -r DEB/requirements.txt
    pip install -r SOAP/requirements.txt

----------------------------------------------------------------------
5. Usage
----------------------------------------------------------------------

Clone the repository and navigate to the supplementary material:

    git clone https://github.com/<your-username>/DEB-SOAP-SM.git
    cd DEB-SOAP-SM/SM_DEB_SOAP

Run DEB feature extraction:

    cd DEB
    python deb.py --input tmid_Si_Savg.csv --output deb_out/

Run supervised classification:

    cd classification
    python 1.py --input tmid_DEB_012labels.csv --model RF

----------------------------------------------------------------------
6. Citation
----------------------------------------------------------------------

If you use this repository, please cite:

Tourani, E. et al. (2025).
"Directional Entropy Bands for Surface Characterization of Polymer Crystallization."
[Polymers].

----------------------------------------------------------------------
7. License
----------------------------------------------------------------------

This repository is released under the MIT License (see LICENSE file).

----------------------------------------------------------------------

