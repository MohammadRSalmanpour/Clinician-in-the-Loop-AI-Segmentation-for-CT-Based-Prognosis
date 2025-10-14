# Clinician-in-the-Loop-AI-Segmentation-for-CT-Based-Prognosis
Click, Predict, Trust: Clinician-in-the-Loop AI Segmentation for CT-Based Prognosis within the Knowledge-to-Action Framework

**Project Overview**

This repository contains the implementation, experimental results, and supplemental materials for our study on deep learning (DL)-based lung cancer segmentation and prognostic modeling. The work evaluates multiple DL architectures within the Knowledge-to-Action (KTA) framework, focusing on geometric accuracy, radiomics reproducibility, and survival prediction across multi-center CT datasets.

**Data Acquisition and Preprocessing**   (Falahati)


**Deep Learning Segmentation Models** (Falahati)


**Statistical Radiomics Analysis** (Alizadeh)


**Machine Learning Classification** 

This module focuses on prognostic modeling using radiomic features derived from both expert and deep learning (DL)–based segmentation masks. A total of 497 standardized radiomic features were extracted via PySERA, normalized, and analyzed under two complementary frameworks: Supervised Learning (SL) and Semi-Supervised Learning (SSL). SL models were trained using five-fold cross-validation on the NSCLC-Radiomics dataset and externally validated on NSCLC-Radiogenomics and LungCT-Diagnosis. SSL leveraged unlabeled datasets through a pseudo-labeling approach, significantly improving generalization across multi-center cohorts.

To handle the high dimensionality of radiomics data, 38 feature selection and embedding methods (e.g., ANOVA F-test, LASSO, PCA, UMAP) were combined with 24 classifiers, including Logistic Regression, Random Forest, XGBoost, LightGBM, SVMs, and ensemble models (Stacking, Voting, Bagging). Model performance was assessed via Accuracy, Precision, Recall, F1-score, Specificity, and ROC-AUC, averaged across folds and test sets.

Results showed that Semi-Supervised Learning consistently outperformed Supervised Learning, with the LASSO–LightGBM–VNet pipeline achieving the best results (Accuracy = 0.88 ± 0.003, F1 = 0.83 ± 0.004, AUC = 0.76 ± 0.048). This analysis also compared DL- and expert-based features to evaluate reproducibility and clinical relevance, supported by physician-in-the-loop review for interpretability. Overall, this stage bridges DL segmentation, radiomics reproducibility, and survival prediction within the Knowledge-to-Action (KTA) framework—emphasizing reliability, adaptability, and clinical trust.


**Physician-in-the-Loop Evaluation** (Alizadeh)



⁉️ Support
For questions, contact:
Mohammad R. Salmanpour: msalman@bccrc.ca
