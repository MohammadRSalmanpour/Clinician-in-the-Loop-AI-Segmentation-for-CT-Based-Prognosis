# Clinician-in-the-Loop-AI-Segmentation-for-CT-Based-Prognosis
Click, Predict, Trust: Clinician-in-the-Loop AI Segmentation for CT-Based Prognosis within the Knowledge-to-Action Framework

**Project Overview**

This repository contains the implementation, experimental results, and supplemental materials for our study on deep learning (DL)-based lung cancer segmentation and prognostic modeling. The work evaluates multiple DL architectures within the Knowledge-to-Action (KTA) framework, focusing on geometric accuracy, radiomics reproducibility, and survival prediction across multi-center CT datasets.

**Data Acquisition and Preprocessing**   

This study utilized a multi-center cohort comprising 999 patients from 12 public TCIA datasets. To ensure the highest quality ground truth, all lesion masks were manually segmented in consensus by two board-certified thoracic radiologists using 3D Slicer and subsequently verified by an independent clinical expert. A standardized preprocessing pipeline was implemented, as detailed in the provided preprocessing.py code. This pipeline performed several key operations: CT volumes were normalized using z-score scaling, and segmentation masks were binarized. Each volume was then resampled and resized to a uniform size of 64×64×64 voxels. A pivotal step involved tumor-centered analysis and cropping, which identified the largest connected component in the mask and extracted a region of interest (ROI) with a 5-voxel 3D margin. This process preserved critical peritumoral context while minimizing irrelevant background, creating standardized, tumor-focused inputs for the deep learning models.

**Preprocessing Requirements**

### Installation

```bash
pip install SimpleITK>=2.2.0 numpy>=1.21.0 pandas>=1.3.0 scipy>=1.7.0

# Run the preprocessing pipeline:

```bash
python preprocessing.py

# Input Data Structure:

train/
├── CT/                 # Original CT images (.nii.gz format)
│   ├── image1.nii.gz
│   ├── image2.nii.gz
│   └── ...
└── Seg/                # Corresponding segmentation masks (.nii.gz format)
    ├── image1.nii.gz
    ├── image2.nii.gz
    └── ...
```

# Important Notes:

- Both CT images and segmentation masks must have identical filenames

- All files must be in .nii.gz format

- Ensure one-to-one correspondence between CT and mask files

- The pipeline will automatically create all necessary output directories


**Deep Learning Segmentation Models** 

Five distinct and complementary 3D deep learning architectures were benchmarked to comprehensively evaluate segmentation performance: 3D Attention U-Net, ResUNet, V-Net, ReconNet, and SAM-Med3D. These models were selected for their diverse inductive biases; for instance, 3D Attention U-Net enhances boundary focus through attention gates, ResUNet uses residual connections for stable training, V-Net is a proven volumetric segmenter, ReconNet offers a lightweight alternative, and SAM-Med3D represents a state-of-the-art foundation model. Each network was trained and validated on ten datasets (~72% training, ~18% validation) and their generalizability was rigorously tested on two independent external datasets.

The preprocessed, cropped CT volumes served as the primary input to these models, enabling a direct and fair comparison of their ability to segment lung lesions from standardized, tumor-focused contexts. The models predicted the corresponding segmentation masks, which were then evaluated against the expert-verified ground truth using a standardized suite of metrics—Dice Similarity Coefficient (Dice), Intersection over Union (IoU), and Hausdorff Distance—computed via the AllMetrics library to ensure reproducibility and eliminate platform-dependent discrepancies. This rigorous benchmarking revealed V-Net as the top-performing model, achieving superior geometric accuracy (Dice: 0.83 ± 0.07) and establishing it as the most robust backbone for our clinician-in-the-loop pipeline.

**Statistical Radiomics Analysis** 

We conducted comprehensive radiomics analysis to validate the clinical reliability of our AI segmentations. A total of 497 IBSI-compliant radiomics features were extracted from both expert and DL-generated masks using PySERA/ViSERA. Feature stability was rigorously assessed through Spearman correlation, ICC, and MANOVA, confirming that our segmentation pipeline preserves prognostically critical information.

Complementing the quantitative radiomics analysis, we conducted an extensive qualitative assessment involving six physicians to evaluate the clinical relevance and trustworthiness of AI-generated masks. Three physicians contributed to manual segmentation for ground truth creation, while three independent external physicians validated the AI-generated masks. The evaluation protocol, detailed in Supplemental File 16 - Survey for Statistical Analysis of qualitative assessments.docx, spanned seven critical domains through 21 carefully designed questions.


**Machine Learning Classification** 

This module focuses on prognostic modeling using radiomic features derived from both expert and deep learning (DL)–based segmentation masks. A total of 497 standardized radiomic features were extracted via PySERA, normalized, and analyzed under two complementary frameworks: Supervised Learning (SL) and Semi-Supervised Learning (SSL). SL models were trained using five-fold cross-validation on the NSCLC-Radiomics dataset and externally validated on NSCLC-Radiogenomics and LungCT-Diagnosis. SSL leveraged unlabeled datasets through a pseudo-labeling approach, significantly improving generalization across multi-center cohorts.

To handle the high dimensionality of radiomics data, 38 feature selection and embedding methods (e.g., ANOVA F-test, LASSO, PCA, UMAP) were combined with 24 classifiers, including Logistic Regression, Random Forest, XGBoost, LightGBM, SVMs, and ensemble models (Stacking, Voting, Bagging). Model performance was assessed via Accuracy, Precision, Recall, F1-score, Specificity, and ROC-AUC, averaged across folds and test sets.

Results showed that Semi-Supervised Learning consistently outperformed Supervised Learning, with the LASSO–LightGBM–VNet pipeline achieving the best results (Accuracy = 0.88 ± 0.003, F1 = 0.83 ± 0.004, AUC = 0.76 ± 0.048). This analysis also compared DL- and expert-based features to evaluate reproducibility and clinical relevance, supported by physician-in-the-loop review for interpretability. Overall, this stage bridges DL segmentation, radiomics reproducibility, and survival prediction within the Knowledge-to-Action (KTA) framework—emphasizing reliability, adaptability, and clinical trust.


**Physician-in-the-Loop Evaluation** (Alizadeh)



⁉️ Support
For questions, contact:
Mohammad R. Salmanpour: msalman@bccrc.ca
