# Machine Learning Project: Calibration of Probabilities

## Project proposal: Assignment 2 
The course develops supervised learning primarily through the lens of risk minimization, where models are trained to minimize a surrogate loss such as the logistic or hinge loss. In this framework, the output of a classifier is often interpreted as a score or a probability, and performance is typically evaluated in terms of classification accuracy or expected loss.

However, in many applications it is not enough to predict the correct label; one also needs reliable probability estimates. A model is said to be calibrated if, among all predictions made with confidence 0.8, roughly 80% are correct. Standard learning algorithms, even when optimized for log-loss, can produce poorly calibrated probabilities, especially when they are overconfident or when the model class is misspecified. For more information, refer to this blog post.

The paper by Niculescu-Mizil and Caruana addresses this gap by systematically studying calibration and proposing simple post-processing methods to improve it. This extends the course material by shifting the focus from prediction accuracy to the quality of uncertainty estimates, and by showing how risk minimization alone does not guarantee well-calibrated outputs. It provides a concrete bridge between theoretical loss functions and practical evaluation metrics.

## Index
1. [Introduction](#introduction)
2. [Data](#data)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Conclusion](#conclusion)

## Introduction

Supervised learning algorithms are typically trained to minimize a prediction risk and are often evaluated in terms of classification accuracy. However, in many applications, correctly predicting the class is not sufficient: reliable estimates of the associated probabilities are also important.

A probabilistic classifier is considered well calibrated when its predicted probabilities reflect the observed frequencies. For example, among observations assigned a probability of 80% to the positive class, approximately 80% should actually belong to that class.

Importantly, minimizing classification risk does not guarantee well-calibrated probability estimates. A model may therefore achieve good classification accuracy while still producing unreliable confidence estimates.

The aim of this project is to evaluate and improve the calibration of probabilistic classifiers, investigating not only whether a model predicts the correct class, but also whether its predicted probabilities provide an accurate representation of its confidence.

Organisation of the project:
```
Calibration_of_Probability/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── breast_cancer_analysis_01.ipynb
│   └── diabetes_analysis_02.ipynb
│
├── scripts/
│   ├── __init__.py
│   ├── calibration.py
│   └── visualization.py
│
└── presentation/
    └── probability_calibration_report.pdf

```

## Data

Two real-world binary classification datasets are used to evaluate probability calibration under different data characteristics and levels of complexity.

### Breast Cancer Wisconsin

Dataset source: `load_breast_cancer` from scikit-learn.

The Breast Cancer Wisconsin dataset contains 569 observations described by 30 numerical features computed from digitized images of breast masses. The target distinguishes between malignant and benign tumors.

This dataset represents the simpler classification problem considered in the project and provides a useful starting point for studying probability calibration.

### Diabetes

Dataset source: `fetch_openml` from scikit-learn (`diabetes`, version 1).

The Diabetes dataset contains 768 observations described by 8 numerical features related to clinical and demographic characteristics. The target indicates whether an individual tested positive or negative for diabetes.

Compared with the Breast Cancer dataset, this dataset presents a more challenging classification problem. These characteristics make the dataset particularly useful for investigating how calibration methods behave in a more complex setting.

## Project Structure

The workflow is organised into five main phases.

1. Data Preparation

   Each dataset is first explored and prepared for the classification task.

   The main preprocessing steps include:

   - exploratory analysis of the variables and target distribution
   - identification of missing or implausible values
   - MICE-style iterative imputation for the Diabetes dataset
   - feature standardization for Logistic Regression

2. Data Splitting

   Each dataset is divided into three independent subsets:

   - Training set (50%) → used to train the original classifiers
   - Calibration set (30%) → used exclusively to fit the calibration methods
   - Test set (20%) → used only for the final evaluation

   The use of a separate calibration set prevents the calibration methods from being fitted on the same observations used to train the original classifier, reducing the risk of biased probability estimates.

3. Classification Models

   Two probabilistic classifiers are trained on the training set:

   - Logistic Regression → a simple and interpretable linear classifier
   - Random Forest → a more flexible ensemble classifier based on multiple decision trees

4. Probability Calibration

   Two calibration methods are implemented from scratch and fitted exclusively on the calibration set:

   - Platt Scaling → learns a sigmoid transformation of the model scores. Its parametric form makes it relatively stable when the amount of calibration data is limited.
   - Isotonic Regression → learns a non-decreasing transformation of the model scores. It is more flexible and can capture complex forms of miscalibration, but it is also more susceptible to overfitting with small calibration sets.

   Calibration try to modifies the probability estimates produced by the classifiers rather than retraining the original models.

5. Evaluation and Calibration Analysis

   The original and calibrated models are evaluated on the independent test set using:

   - Accuracy → proportion of correctly classified observations
   - Log-loss → quality of the predicted probabilities, with a strong penalty for confident incorrect predictions
   - Brier score → squared difference between predicted probabilities and observed outcomes

   Reliability diagrams are also used to visually assess calibration. A well-calibrated classifier should produce a curve close to the diagonal, indicating agreement between predicted probabilities and observed frequencies.

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The analyses can then be reproduced by running the notebooks contained in the `notebooks/` directory.

## Conclusion

This project demonstrates the importance of evaluating probabilistic classifiers not only in terms of classification accuracy, but also according to the reliability of their probability estimates.

The analysis shows that calibration can improve probability estimates without necessarily improving classification accuracy. Across the two datasets and classifiers, Platt Scaling provides the most consistent improvements in Log-loss and Brier score, offering stable probability corrections. Isotonic Regression, while more flexible, shows a greater tendency to overfit when the calibration set is limited, sometimes producing excessively confident probability estimates.

Overall, the results highlight that good classification performance does not automatically imply well-calibrated probabilities. Among the calibration approaches considered, Platt Scaling represents the most robust solution in these experiments, providing a good trade-off between calibration performance, stability and model complexity.
