# Machine Learning Models: Supervised and Unsupervised Learning

This repository provides a comprehensive implementation and analysis of key supervised and unsupervised machine learning algorithms using Python and `scikit-learn`. It includes clustering, dimensionality reduction, classification, and regression models, along with visualizations, performance metrics, and reproducible workflows. This repository was created in fulfillment of the requirements for CMOR 438 SPR 2026.

---

## Overview

### Unsupervised Learning

Unsupervised learning algorithms discover hidden patterns or structures in unlabeled data. These models do not rely on predefined outputs and are useful for tasks such as clustering, anomaly detection, and dimensionality reduction.

#### Key Concepts:
- Clustering: Grouping similar data points based on features.
  - K-Means Clustering
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Dimensionality Reduction: Reducing feature space while preserving variance.
  - Principal Component Analysis (PCA)
  - Image Compression with Singular Value Decomposition (SVD)

#### Challenges:
- Lack of ground truth for evaluation
- Model interpretability
- Choosing the right algorithm for structure discovery

---

### Supervised Learning

Supervised learning relies on labeled data to train models for classification or regression tasks. These models learn a mapping between inputs and known outputs.

#### Key Concepts:
- Classification: Predicting categorical labels.
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Trees & Random Forests
  - Neural Networks
- Regression: Predicting continuous outcomes.
  - Linear Regression
  - Random Forest Regression
  - Boosted Trees (AdaBoost, Gradient Boosting)

#### Workflow:
1. Training: Model learns from labeled data by minimizing a loss function.
2. Testing: Evaluated on unseen data to assess generalization performance.

#### Benefits and Limitations:
- High accuracy with sufficient data
- Clear performance metrics (e.g., accuracy, MSE, F1-score)
- Requires large labeled datasets
- Risk of overfitting on small or noisy datasets

---

## Datasets and Use Cases

This project uses several standard datasets to evaluate different learning techniques across tasks:

| Dataset | Source | Used For | Description |
|--------|--------|----------|-------------|
| Wine | `sklearn.datasets.load_wine` | K-Means, PCA | 13 chemical properties of wine from 3 cultivars |
| California Housing | `sklearn.datasets.fetch_california_housing` | DBSCAN, Decision Trees | Housing prices and socio-economic data across California |
| MNIST | `sklearn.datasets.fetch_openml` | SVD Compression | 28×28 grayscale digit images for visual compression |
| LFW Faces | `sklearn.datasets.fetch_lfw_people` | SVD Compression | Grayscale face images of public figures |
| Breast Cancer | `sklearn.datasets.load_breast_cancer` | Classification | Tumor features used to classify benign vs. malignant |
| Fashion MNIST | `tensorflow.keras.datasets` | Neural Networks | 28×28 grayscale images of clothing for image classification |
| CDC Obesity | CDC database | Linear Regression | Physical activity and obesity rates across U.S. states |
| Digits Dataset | `sklearn.datasets.load_digits` | KNN | 8×8 grayscale digit images used for classification |

---

## Reproducibility

To reproduce results in this repository, please reference 'README.md' files in the respective supervised and unsupervised sections.
