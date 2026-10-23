# ChurnIQ — Telco Customer Churn Prediction

An interactive machine learning application that estimates customer churn probability and helps prioritize customer retention efforts using **XGBoost, Python, and Streamlit**.

ChurnIQ transforms customer subscription, service, and billing information into actionable predictions through an interactive dashboard. It also provides a customer prioritization workflow to help explore which customers may need retention outreach first.

<p align="center">
  <a href="https://churniq-telco-customer-churn.streamlit.app">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-Machine%20Learning-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Model-XGBoost-EC6B23?style=for-the-badge" alt="XGBoost">
</p>

---

## Table of Contents

* [Project Overview](#project-overview)
* [Business Problem](#business-problem)
* [Key Features](#key-features)
* [Application Screenshots](#application-screenshots)
* [Machine Learning Workflow](#machine-learning-workflow)
* [Model Evaluation](#model-evaluation)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Installation and Setup](#installation-and-setup)
* [How to Use](#how-to-use)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)

---

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Identifying customers who may be at risk of leaving can help businesses investigate potential issues and make more informed retention decisions.

**ChurnIQ** is an interactive web application built around an XGBoost binary classification model. Users can enter a customer profile, explore the model's churn prediction, and organize customer profiles by estimated churn probability.

The application works with a dataset containing **7,032 customer records**.

### Project Objectives

* Estimate the probability that a customer will churn.
* Make machine learning predictions accessible through an interactive web interface.
* Explain model evaluation metrics in a business context.
* Support customer prioritization for retention campaign planning.
* Demonstrate an end-to-end applied machine learning project.

---

## Business Problem

Customer acquisition can be costly, making customer retention an important business objective.

However, not every customer has the same likelihood of leaving. A churn prediction model can help identify patterns associated with churn and support decisions about which customers deserve further attention.

ChurnIQ addresses this problem by combining customer profile inputs, a trained classification model, and an interactive prioritization workflow.

> **Important:** A predicted churn probability is an estimate, not a guarantee that a customer will leave.

---

## Key Features

### 1. Overview Dashboard

Provides a high-level introduction to the project, its business problem, and the prediction workflow.

### 2. Individual Churn Prediction

Enter customer subscription, service, and billing information to obtain:

* A predicted churn outcome.
* An estimated churn probability.
* A visual representation of the prediction.
* A probability-based risk interpretation.

The application prepares the input features and applies the saved preprocessing artifacts before passing them to the XGBoost model.

### 3. Retention Campaign Prioritization

Explore customer profiles and prioritize potential retention outreach based on estimated churn probability.

Features include:

* Choose the number of customer profiles to evaluate.
* Add and edit customer profiles.
* Recalculate predictions after editing a profile.
* Organize customers into priority groups.
* Export the prioritized customer list as a CSV file.

The priority thresholds are configurable business rules used to organize outreach. They are not independently validated business outcomes.

### 4. Model Insights

Displays the model's recorded evaluation metrics and explains how to interpret them, including the trade-off between identifying customers who churn and avoiding unnecessary retention outreach.

### 5. Dataset Metadata

Provides descriptions of dataset fields, their possible values, and their roles in the model.

---

## Application Screenshots

Screenshots will be added here to demonstrate the application's interface and main workflows.

### Overview Dashboard

![ChurnIQ Overview Dashboard](assets/overview.png)

### Churn Prediction

![ChurnIQ Churn Prediction](assets/churn-prediction.png)

### Retention Campaign

![ChurnIQ Retention Campaign](assets/retention-campaign.png)

### Model Insights

![ChurnIQ Model Insights](assets/model-insights.png)

---

## Machine Learning Workflow

The application's prediction workflow follows these steps:

1. **Customer input:** Collect the customer's subscription, service, and billing attributes.
2. **Feature preparation:** Prepare the input features in the format expected by the trained model.
3. **Preprocessing:** Apply the saved scaler and use the saved feature-column definition.
4. **Model inference:** Pass the prepared feature vector to the trained XGBoost classifier.
5. **Probability estimation:** Obtain the model's estimated probability for the churn class.
6. **Result presentation:** Display the prediction and estimated probability in the application.
7. **Retention prioritization:** Use estimated probabilities to organize customer profiles for further review.

### Saved Model Artifacts

The application loads its trained model and preprocessing artifacts from the `XGBoost Model` directory:

* `xgb_churn_model.pkl` — trained XGBoost model.
* `scaler.pkl` — saved feature scaler.
* `feature_columns.pkl` — saved feature-column definition.

These artifacts allow the application to use the existing trained model without retraining it each time the app starts.

---

## Model Evaluation

The application's Model Insights page currently reports the following test-set results:

| Metric          | Reported Result |
| --------------- | --------------: |
| Accuracy        |             73% |
| Churn Precision |             49% |
| Churn Recall    |             80% |
| Churn F1-score  |             61% |

### Interpretation

* **Accuracy:** The proportion of all test examples classified correctly.
* **Precision:** Of the customers predicted to churn, the proportion who actually churned.
* **Recall:** Of the customers who actually churned, the proportion identified by the model.
* **F1-score:** A combined measure of precision and recall.

The reported recall suggests that the model identifies a substantial proportion of churn cases in the evaluated test set. The lower precision indicates that some customers predicted to churn may not actually churn.

The appropriate balance depends on the business cost of missed churn cases compared with the cost of unnecessary retention outreach.

**Evaluation note:** These are the results currently recorded in the application. They should be reconfirmed against the finalized preprocessing pipeline and evaluation code before being treated as independently verified model performance.

---

## Technology Stack

| Technology   | Purpose                                      |
| ------------ | -------------------------------------------- |
| Python       | Core programming language                    |
| Pandas       | Data manipulation and preparation            |
| NumPy        | Numerical operations                         |
| Scikit-learn | Preprocessing and machine learning utilities |
| XGBoost      | Churn classification model                   |
| Streamlit    | Interactive web application                  |
| Plotly       | Interactive visualizations                   |
| Joblib       | Loading saved model artifacts                |

SQL was also used as part of the broader data analysis workflow described in the application.

---

## Project Structure

```text
ChurnIQ-Telco-Customer-Churn/
│
├── app.py
├── requirements.txt
│
├── XGBoost Model/
│   ├── xgb_churn_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── assets/
│   ├── overview.png
│   ├── churn-prediction.png
│   ├── retention-campaign.png
│   └── model-insights.png
│
├── .gitignore
└── README.md
```

The `assets/` directory is used for the README screenshots. Keep the actual application and model filenames consistent with the files in your repository.

---

## Installation and Setup

### Prerequisites

* Python installed on your machine.
* Git installed on your machine.
* The project repository and required model artifacts.

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ChurnIQ-Telco-Customer-Churn
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open it in your browser to interact with the application.

---

## How to Use

1. Open the deployed application or run it locally.
2. Navigate to **Overview** to learn about the project.
3. Open **Churn Prediction** and enter a customer profile.
4. Review the predicted churn outcome and estimated probability.
5. Open **Retention Campaign** to evaluate and prioritize customer profiles.
6. Download the prioritized customer list as a CSV file if needed.
7. Visit **Model Insights** to review the recorded evaluation metrics.
8. Use **Metadata** to understand the dataset fields and their meanings.

---

## Limitations

* Predictions depend on the quality and representativeness of the data used to train the model.
* Estimated churn probabilities are model outputs and should not be interpreted as certain future outcomes.
* Retention priority thresholds are illustrative business rules, not proof of campaign effectiveness.
* The reported evaluation metrics should be validated against the finalized training and preprocessing workflow.
* The application is an educational portfolio project and is not a substitute for a production customer analytics system.

---

## Future Improvements

Potential next steps include:

* Reproducible model training and evaluation scripts.
* A documented cross-validation and hyperparameter-tuning workflow.
* More detailed model explainability using feature importance or SHAP.
* Monitoring model performance as customer behavior changes.
* Business-focused evaluation of retention strategies.
* Automated tests for preprocessing and prediction consistency.

---

## Author

**Mohamed Elshafie**

Computer Science student interested in Data Science, Machine Learning, and AI engineering.

---

## Disclaimer

ChurnIQ is an educational portfolio project developed to demonstrate an applied machine learning workflow. Predictions and prioritization results are intended for exploration and decision support, not as guarantees of customer behavior or retention campaign outcomes.
