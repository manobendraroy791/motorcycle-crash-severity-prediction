# Motorcycle Crash Severity Prediction: A First-Principles ML Approach

## Overview
This repository contains a Logistic Regression classification engine built entirely from scratch using pure **NumPy** and mathematics. 

The objective of this project is to predict the severity of motorcycle crashes (Minor/Moderate vs. Severe/Fatal) based on variables such as speed, rider experience, road conditions, and helmet usage. 

Instead of relying on high-level, pre-built machine learning libraries (like Scikit-Learn), this project implements the core mathematical architecture—including vectorization, the sigmoid activation function, log-loss cost calculation, and gradient descent—from first principles.

## The Mathematical Architecture
To ensure maximum computational efficiency and a deep understanding of the underlying mechanics, the engine features:
* **Custom One-Hot Encoding:** Categorical variables (road type, weather, alcohol influence) are flattened into binary matrices.
* **Feature Scaling:** Continuous variables (speed, experience) are standardized using Z-score normalization to ensure a smooth gradient descent landscape.
* **Vectorized Gradient Descent:** The forward pass ($z = wX + b$) and backward pass (calculating derivatives $dw$ and $db$) are computed across all training instances simultaneously using matrix dot products (`np.dot`), avoiding computationally expensive `for` loops.

## Dataset
The model was trained on a dataset of 5,000 recorded motorcycle crashes. 
* **Target Variable:** `severity` (Mapped to 0: Minor/Moderate, 1: Severe/Fatal)
* **Features:** 14 independent variables (after encoding and dropping redundant columns to avoid the dummy variable trap).

## Model Performance
The dataset was randomly shuffled and split into an 80% training set and a 20% holdout test set to ensure generalizability.

* **Training Accuracy:** 80.3%
* **Test Accuracy:** 81.4%

*Note: The model demonstrates exceptional generalization, with the test accuracy slightly outperforming the training accuracy, indicating a highly robust fit with zero overfitting to the training noise.*

## Core Technologies
* `Python`
* `NumPy` (Matrix operations, linear algebra, calculus implementation)
* `Pandas` (Initial data ingestion and parsing)
* `Matplotlib` (Learning curve visualization)

## How to Run
1. Clone the repository.
2. Ensure the `Motorcycle_crash_data.xlsx` file is in the root directory.
3. Run the complete pipeline in the provided Jupyter Notebook (`crash_severity_model.ipynb`). The script will output the final log-loss cost, test accuracy, and plot the gradient descent learning curve.
