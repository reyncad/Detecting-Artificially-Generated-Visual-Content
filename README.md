# Detecting-Artificially-Generated-Visual-Content

## 1. PROJECT OVERVIEW 

This project presents a critical analysis of deep learning (DL) and machine learning (ML) architectures applied to the detection of AI-generated synthetic images (Deepfakes). In response to the growing societal problem of **disinformation** and media manipulation, the core aim is to identify the most robust, reliable, and cost-effective model suitable for practical deployment.

## 2. ORIGINAL CONTRIBUTION AND NOVELTY

The novelty of this work lies not in proposing a single new model, but in the **rigorous, multi-faceted comparative framework** used to evaluate existing state-of-the-art architectures.

**A. Quantiﬁcation of Performance vs. Cost (The Trade-off):**
We quantify the true computational cost of classification by directly comparing high-accuracy kernel-based models (SVM) with high-speed ensemble models (XGBoost).
* **Key Finding:** The project demonstrates that while SVM achieves a marginal accuracy gain, it costs $\mathbf{\sim 24 \times}$ more time to train ($\approx 114$ minutes) than XGBoost ($\approx 4.7$ minutes). This provides empirical evidence for the **practical cost/benefit ratio** for real-time systems.
* **Supporting Evidence:** The **PCA-based Feature Selection** ablation further proves this commitment by showing that the SVM training time can be reduced by $\mathbf{29.5\%}$ while maintaining near-identical F1-scores.

**B. Robustness Analysis Against Real-World Distortion:**
We directly address a common gap in the literature (vulnerability to noise and compression) by performing a novel stress test on the champion model (SVM\_RBF).
* **Key Finding:** The model shows high resilience to **Gaussian Blur** and **JPEG Compression ($\approx 86\% - 90\%$ F1)**, but exhibits a catastrophic failure ($\mathbf{46.51\%}$ F1) when exposed to **Gaussian Noise ($\mathbf{std=0.1}$)**.
* **Implication:** This robustly quantifies the **Achilles’ heel** of the hybrid architecture, serving as a direct foundation for future noise-filtering research.

**C. Methodological Rigor (Reliability):**
All results are validated through $\mathbf{5-Fold \ Stratified \ Cross-Validation}$ and reported with **Mean $\pm$ Standard Deviation**, ensuring that the findings are not stochastic but scientifically consistent and reliable.

## 3. EXPERIMENTAL SETUP AND METHODOLOGY

The analysis compares three distinct groups of classification models:

* **Group A (Hybrid ML - Our Focus):** **DL (ResNet50) $\to$ ML Classifier.** These models were trained on features extracted from the DL backbone. (Models: SVM, XGBoost, Random Forest, LogReg, KNN).
* **Group B (Pure DL - Teammate's Focus):** End-to-end models (ConvNeXt-Tiny, EfficientNetB0, DeiT-Tiny, etc.) trained on raw pixels.
* **Feature Extraction:** All images were processed by a pre-trained **ResNet50** (to extract 2048-dimensional vectors) or used directly for end-to-end DL models.

## 4. CODE AND OUTPUT FILES

All experiments were executed on **Google Colaboratory Pro (NVIDIA A100/L4 GPU)** using PyTorch and Scikit-learn.

| Output File Name | Content | Report Section |
| :--- | :--- | :--- |
| `Hibrit_kfold_ozet_sonuclari.txt` | Mean and Std Dev of all 5 ML models (ACC, F1, AUC). | Comprehensive Performance (Table 3) |
| `Hibrit_FS_Kaniti_Raporu.txt` | Quantitative proof of the $\mathbf{29.5\%}$ training time reduction using PCA. | Ablation/Feature Selection Analysis |
| `Hibrit_Dayaniklilik_Raporu.txt` | F1 scores across distorted data (JPEG, Noise, Blur). | Robustness/Praticality Analysis |
| `Hibrit_roc_curve.png` | Final model ROC curve (Görsel Kanıt). | Section 1.6.3 (ROC Curves) |
