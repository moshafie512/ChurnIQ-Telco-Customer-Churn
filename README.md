# ChurnIQ | Customer Churn Prediction

**ChurnIQ** is an interactive machine learning project designed to predict customer churn and help businesses identify customers who are likely to leave. The application combines data analysis, churn prediction, and retention campaign prioritization in one dashboard.

## Overview

Customer churn is a major challenge for subscription-based businesses. ChurnIQ uses machine learning to analyze customer data, predict churn risk, and support data-driven customer retention decisions.

## Key Features

* **Exploratory Data Analysis:** Explore customer churn patterns and relationships within the dataset.
* **Churn Prediction:** Enter customer information to predict whether a customer is likely to churn and view the estimated probability.
* **Retention Campaign:** Manage customer profiles, prioritize customers for retention efforts, and export results to CSV.
* **Model Insights:** Review model evaluation metrics and explore feature importance.
* **Interactive Dashboard:** Access the project's functionality through a user-friendly Streamlit web application.

## Visual Showcase

### 1. Exploratory Data Analysis

An overview of customer churn patterns in the dataset.

![Customer Churn Analysis](assets/customer-churn.png)

### 2. Feature Importance

The top 15 features identified by the tuned XGBoost model.

![Top 15 Feature Importances](assets/top-15-feature-importances-tuned-XGBoost.png)

### 3. Application Dashboard

The ChurnIQ overview dashboard.

![ChurnIQ Overview](assets/overview.png)

### 4. Prediction Results

An example of the model's customer churn prediction.

![Churn Prediction Result](assets/prediction-result.png)

## Machine Learning Workflow

1. Explore and analyze the customer dataset.
2. Prepare the data for machine learning.
3. Train and tune an XGBoost classification model.
4. Evaluate model performance using classification metrics.
5. Integrate the trained model into an interactive Streamlit application.

## Technologies Used

* **Python** — Core programming language
* **Pandas & NumPy** — Data manipulation and analysis
* **Scikit-learn** — Data preprocessing and model evaluation
* **XGBoost** — Churn classification model
* **Streamlit** — Interactive web application
* **Plotly & Matplotlib** — Data visualization

## Installation & Usage

To run ChurnIQ locally:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ChurnIQ
pip install -r requirements.txt
streamlit run app.py
```

Make sure the trained model files are available in the expected project directory.

## Limitations

* Predictions depend on the quality and characteristics of the dataset used to train the model.
* Feature importance indicates which variables the model relies on; it does not necessarily imply causation.
* Predictions should support business decisions rather than replace human judgment.

## Future Improvements

* Experiment with additional machine learning models.
* Improve model performance through further tuning and feature engineering.
* Expand retention strategies with more customer-specific recommendations.

## Author

**Mohamed Elshafie**
AI & Data Science Student | Machine Learning & Data Science
