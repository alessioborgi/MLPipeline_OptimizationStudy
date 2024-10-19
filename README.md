# MLPipeline: Multivariate Anomaly Detection in High-Dimensional Datasets

## Overview
This repository contains the implementation and analysis of machine learning techniques for multivariate anomaly detection in high-dimensional datasets. The project focuses on exploring various preprocessing methods, dimensionality reduction techniques, and classification algorithms in scenarios with different computational power constraints.

## Contents
- [Introduction](#introduction)
- [Preprocessing](#preprocessing)
- [Algorithms Analysis](#algorithms-analysis)
- [Results](#results)
- [Conclusions](#conclusions)
- [Future Works](#future-works)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contact](#contact)

## Introduction
The purpose of this project is to analyze and compare various machine learning algorithms for multivariate anomaly detection in high-dimensional datasets. The repository contains experiments on two datasets, employing a range of preprocessing steps and classification models. It also presents solutions for real-world scenarios with limited computational power.

## Preprocessing
The preprocessing phase is crucial for ensuring consistent and unbiased results across all algorithms and datasets. The key preprocessing steps include:

1. **Missing Values Handling**: This project did not require handling missing values as both datasets were complete.
2. **Data Standardization**: Applied a standard scaler to normalize features to a mean of 0 and a standard deviation of 1.
3. **Dimensionality Reduction**: Utilized Principal Component Analysis (PCA) and T-Distributed Stochastic Neighbor Embedding (T-SNE) to reduce the dataset dimensions while retaining 99.9% of cumulative explained variance.
4. **Data Imbalance**: Both datasets were perfectly balanced, eliminating the need for resampling techniques.
5. **Train-Test Split**: Implemented an 80-20 split using Stratified Shuffle Split to maintain class distribution.

## Algorithms Analysis
The project investigates a variety of machine learning algorithms, including:
- **Random Forest**
- **Kernelized Support Vector Machine (SVM)**
- **Logistic Regression**
- **XGBoost**
- **LightGBM**
- **CatBoost**

### Grid Search and Cross-Validation
Each algorithm underwent extensive hyperparameter tuning using Grid Search Cross-Validation to identify the best model settings.

### Voting Ensemble
To enhance the robustness of model predictions, a Weighted Hard Voting ensemble strategy was employed, combining the top-performing algorithms based on their accuracy scores.

## Results
The project evaluates models based on three performance dimensions:
1. **Metrics Performance**: Accuracy was the primary metric, supplemented by precision, recall, and F1-score for comprehensive evaluation.
2. **Time Complexity**: Focused on training time, using a normalization method to ensure fair comparison across different algorithms.
3. **Hardware Complexity**: Analyzed model sizes to gauge their feasibility for real-world deployment, particularly in resource-constrained environments.

### Dataset 1: High Computational Power Solution (PCA-Based)
- Best Algorithm: Logistic Regression with an accuracy of 98.93%
- Ensemble Method: Weighted Hard Voting for enhanced decision-making
- Low Computational Power Solution: T-SNE + Logistic Regression achieved 98.33% accuracy

### Dataset 2: High Computational Power Solution (PCA-Based)
- Best Algorithm: Logistic Regression with an accuracy of 97.35%
- Ensemble Method: Weighted Hard Voting for robustness
- Low Computational Power Solution: T-SNE + Logistic Regression achieved 96.86% accuracy

## Conclusions
The analysis across two datasets reveals that Logistic Regression consistently outperforms other models in terms of accuracy and hardware complexity. For time efficiency, XGBoost and Random Forest emerged as strong candidates. The use of dimensionality reduction techniques like PCA and T-SNE proved effective, particularly in low-computational power scenarios.

## Future Works
- Integration of advanced dimensionality reduction methods such as Autoencoders.
- Exploration of additional ensemble learning strategies like Stacking and Gradient Boosting ensembles.
- Expansion to real-time anomaly detection using the developed ML pipeline.

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/ML_Pipeline_for_Anomaly_Detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ML_Pipeline_for_Anomaly_Detection
    ```
3. Install the required dependencies (see [Requirements](#requirements) below).
4. Run the preprocessing and model training scripts:
    ```bash
    python scripts/train_model.py
    ```

## Requirements
- Python 3.x
- Required Python packages (install using `requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```
- Packages include:
  - NumPy
  - Pandas
  - Scikit-learn
  - XGBoost
  - LightGBM
  - CatBoost
  - Matplotlib
  - Seaborn

## Contact
For any questions or feedback, please contact:  
**Alessio Borgi**  
Email: borgi.1952442@studenti.uniroma1.it
