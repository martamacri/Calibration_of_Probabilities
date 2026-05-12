# Project_ML

## Assignment 2 — Calibration of Probabilities

The course develops supervised learning primarily through the lens of risk minimization, where models are trained to minimize a surrogate loss such as the logistic or hinge loss. In this framework, the output of a classifier is often interpreted as a score or a probability, and performance is typically evaluated in terms of classification accuracy or expected loss.
However, in many applications it is not enough to predict the correct label; one also needs reliable probability estimates. A model is said to be calibrated if, among all predictions made with confidence 0.8, roughly 80% are correct. Standard learning algorithms, even when optimized for log-loss, can produce poorly calibrated probabilities, especially when they are overconfident or when the model class is misspecified. For more information, refer to this blog post.
The paper by Niculescu-Mizil and Caruana addresses this gap by systematically studying calibration and proposing simple post-processing methods to improve it. This extends the course material by shifting the focus from prediction accuracy to the quality of uncertainty estimates, and by showing how risk minimization alone does not guarantee well-calibrated outputs. It provides a concrete bridge between theoretical loss functions and practical evaluation metrics.

Objective
Evaluate and improve the calibration of probabilistic classifiers
Dataset
Two real-world classification datasets
Tasks
Train:
Logistic Regression
Random Forest
Apply:
Platt scaling
Isotonic regression
Evaluate:
Accuracy
Log-loss
Brier score
Plot:
Reliability diagrams
Expected Output
Calibration curves before and after correction
Quantitative comparison of metrics
Discussion of miscalibration


Apprendimento supervisionato per minimizzare il rischio --> l'output della classificazione è una percentuale (valutate in termini di accuratezza di classificazione)
Servono stime di probabilità affidabili --> modello calibrato se 
