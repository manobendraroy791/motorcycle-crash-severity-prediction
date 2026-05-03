# Motorcycle Crash Severity Prediction (Built from Scratch in NumPy)

## 📌 Project Overview
This project involves the development of a custom Logistic Regression machine learning engine, built entirely from scratch using pure NumPy and linear algebra. The model predicts the severity of motorcycle crashes (Minor vs. Severe/Fatal) based on a dataset of 5,000 crash records. 

To demonstrate a deep understanding of core machine learning mathematics, **no high-level ML libraries (like Scikit-Learn) were used for the predictive engine.** The gradient descent optimization, cost function (log-loss), and polynomial feature expansions were all mathematically derived and hard-coded.

## 🛠️ Methodology & Architecture
* **Data Preprocessing:** Handled categorical variables using One-Hot Encoding and scaled continuous variables (Speed, Experience) using custom Z-score normalization to ensure gradient descent convergence.
* **The Engine:** A vectorized Logistic Regression model trained via standard Gradient Descent.
* **Evaluation:** Split data into an 80/20 Train/Test set to evaluate performance on unseen data, plotting cost-reduction over 2,000+ epochs to monitor the learning curve.

## 📊 Results & Feature Importance Experiment
Two distinct models were built and tested within the laboratory notebook to prove the necessity of contextual data in traffic engineering:

### 1. The 17-Feature Contextual Model
* **Features Used:** Speed, Rider Experience, Helmet Usage, Road Condition, Weather, Alcohol Influence, etc.
* **Dimensions:** Data mapped to a 16-dimensional hyperplane.
* **Test Accuracy:** **81.4%**

### 2. The 2-Feature Kinematic Model (A/B Test)
An experiment was conducted to see if kinematic data (Speed) and rider skill (Experience) alone could predict fatalities. The model was reduced to 2 features, allowing the decision boundary to be visualized. 
* **Polynomial Expansion:** To give the model the best chance, $X_1^2$, $X_2^2$, and $X_1X_2$ features were synthesized to allow the math to draw complex, non-linear boundaries (curves and parabolas).
* **Test Accuracy:** Collapsed to **64.5%**
* **Conclusion:** This mathematically proves that crash severity cannot be accurately predicted by speed and rider skill alone. The high accuracy of the 17-feature model relies heavily on environmental context (Weather, Road) and safety compliance (Helmets, Alcohol).

## 🚀 How to Run the Code
1. Clone this repository or download the `.ipynb` and `.xlsx` files.
2. Open `crash_severity_model.ipynb` in **Google Colab** (recommended) or a local Jupyter Notebook environment.
3. Run the notebook from the top. 
4. When prompted by the first cell, upload the `Motorcycle_crash_data.xlsx` file. 
5. The pipeline will automatically preprocess the data, run the gradient descent loops, output the final weights, and plot the dual-monitor telemetry dashboards.

## 💻 Tech Stack
* **Python** (Pandas for data ingestion, NumPy for matrix calculus, Matplotlib for data visualization)
