# Motorcycle Crash Severity Prediction: From Linear Boundaries to Advanced Ensembles

## 📌 Project Overview
This project explores the mathematical architectures of machine learning algorithms to predict the severity of motorcycle crashes (Minor vs. Severe/Fatal) based on a dataset of 5,000 crash records. 

The project is divided into two distinct engineering phases:
1. **Phase 1: The Foundation.** Building a custom Logistic Regression engine entirely from scratch using pure NumPy and matrix calculus to prove an understanding of gradient descent and parametric modeling.
2. **Phase 2: The Upgrade.** Transitioning to advanced non-linear architectures (Decision Trees, Random Forests, and XGBoost) to solve a life-critical engineering flaw discovered in Phase 1: unacceptable False Negative rates.

## 🛠️ Phase 1: Logistic Regression (Built from Scratch in NumPy)
To demonstrate a deep understanding of core machine learning mathematics, **no high-level ML libraries were used for the initial predictive engine.** The gradient descent optimization, cost function (log-loss), and polynomial feature expansions were all mathematically derived and hard-coded.

### The Feature Importance Experiment
Two distinct models were built and tested to prove the necessity of contextual data in traffic engineering:

* **1. The 17-Feature Contextual Model:** Mapped data to a 16-dimensional hyperplane (Speed, Helmet Usage, Road Condition, Alcohol, etc.). Achieved a test accuracy of **81.4%**.
* **2. The 2-Feature Kinematic Model (A/B Test):** Reduced to only Speed and Rider Experience. Synthesized $X_1^2$, $X_2^2$, and $X_1X_2$ features to allow the math to draw complex, non-linear boundaries. Test accuracy collapsed to **64.5%**.
* **Conclusion:** This mathematically proves that crash severity cannot be accurately predicted by kinematics alone. High accuracy relies heavily on environmental context and safety compliance.

### The Parametric Flaw (Why Phase 2 was Necessary)
While the 17-feature Logistic Regression model achieved decent overall accuracy, an analysis of the Confusion Matrix revealed a critical flaw for safety engineering: **High False Negatives (predicting a fatal crash as minor).** Because Logistic Regression relies on a rigid, continuous equation, it struggled to capture complex edge cases. Even when the decision threshold was aggressively lowered to `y >= 0.20` to favor predicting fatalities, the model still allowed too many life-threatening crashes to slip through as "Minor."

## 🚀 Phase 2: Advanced Non-Linear Architectures
To eliminate the False Negatives, the pipeline was upgraded to non-parametric, tree-based algorithms. These models do not rely on continuous equations; instead, they calculate Impurity Reduction (Information Gain) to find hidden, highly specific thresholds.

* **Decision Tree (The Baseline):** Replaced the linear equation with a hierarchical flowchart. Successfully captured non-linear thresholds but proved slightly brittle (high variance).
* **Random Forest (Bagging):** Engineered a parallel committee of 100 trees using Bootstrapping and Feature Randomness. This averaged out independent errors, dropping False Negatives significantly without overfitting.
* **XGBoost (Boosting):** Deployed a sequential chain of shallow trees. Instead of parallel voting, every new tree in the XGBoost chain was strictly trained to target the *continuous residual errors* of the previous tree, systematically hunting down the remaining False Negatives.

## 📊 Final Model Comparison
The following table outlines the performance shift from a single Decision Tree to advanced ensembles. *(Note: The primary metric of success for this system is not overall accuracy, but the minimization of life-critical False Negatives).*

| Algorithm Architecture | Training Accuracy | Testing Accuracy | False Negatives (Missed Fatalities) |
| :--- | :--- | :--- | :--- |
| **Decision Tree** (Max Depth: 6) | [90.85 %] | [89.5 %] | [49] |
| **Random Forest** (200 Trees, Depth 7) | [91.65 %] | [89.2 %] | [25] |
| **XGBoost** (Sequential Gradient Boosting) | [91.8 %] | [90.1 %] | [43] |

**<img width="1790" height="495" alt="download (22)" src="https://github.com/user-attachments/assets/1dc65770-2b4e-49ec-832a-18583d69518d" />**

## ⚙️ How to Run the Code
1. Clone this repository to your local machine.
2. Open the respective `.ipynb` files in **Google Colab** (recommended) or a local Jupyter Notebook environment.
    * `Motorcycle_crash_data.ipynb` (Phase 1)
    * `crash_severity_Ensembles.ipynb` (Phase 2)
3. Run the notebooks from the top. 
4. When prompted, upload the `Motorcycle_crash_data.xlsx` file. 
5. The pipelines will automatically preprocess the categorical data, run the training sequences, and output the final confusion matrices and comparisons.

## 💻 Tech Stack
* **Mathematics & Engine:** NumPy, SciPy
* **Data Pipelines & Viz:** Pandas, Matplotlib, Seaborn
* **Advanced Ensembles:** Scikit-Learn, XGBoost
