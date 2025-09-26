<h1> SHAPing Latent Spaces in Facial Attribute Classification Models </h1>

Presentation delivered at the BIOSIG 2025 Conference:

https://github.com/user-attachments/assets/9d0dd3c4-35a3-46ce-8a8f-b15f093a2ed5

---


## Overview
This study investigates the use of **SHAP (SHapley Additive exPlanations)** values as an **explainable artificial intelligence (xAI) technique** applied on a **facial attribute classification task**. We analyse the consistency of SHAP value distributions across diverse classifier architectures that share the same feature extractor, revealing that key features driving attribute classification remain stable regardless of classifier architecture. Our findings highlight the challenges in interpreting SHAP values at the individual sample level, as their reliability depends on the model’s ability to learn distinct class-specific features; models exploiting inter-class correlations yield less representative SHAP explanations. Furthermore, pixel-level SHAP analysis reveals that superior classification accuracy does not necessarily equate to meaningful semantic understanding; notably, despite FaceNet exhibiting lower performance than CLIP, it demonstrated a more nuanced grasp of the underlying class attributes. Finally, we address the computational scalability of SHAP, demonstrating that *KernelExplainer* becomes infeasible for high-dimensional pixel data, whereas *DeepExplainer* and *GradientExplainer* offer more practical alternatives with trade-offs. Our results suggest that **SHAP is most effective for small to medium feature sets or tabular data**, providing interpretable and computationally manageable explanations.
# Table of Contents
**[Installation Instructions](#dataset)**<br>


## Dataset
- BUPT-Balancedface dataset: ~1.3M images, balanced ethnicity groups.
- Facial attributes from Neto et al.'s annotations with 47 labels.

## Installation
```bash
pip install -r requirements.txt
```
