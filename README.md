# Telco Customer Churn Prediction 🚀

A supervised machine learning pipeline designed to analyze customer behavior and predict churn in the telecommunications industry using the **Telco Customer Churn** dataset.

The project handles raw feature cleaning, correlation-based feature selection, imbalance management via **SMOTE**, feature scaling, polynomial interactions, and model evaluation across multiple classifiers.

---

## 📊 Dataset Overview

The project uses the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset, which contains customer demographic details, subscribed services, account details, and churn status.

- **Target Variable:** `Churn` (`0` = No Churn, `1` = Churn)
- **Key Features Selected:** Features with a strong absolute correlation ($> 0.15$) with `Churn`, including:
  - `Contract`, `tenure`, `OnlineSecurity`, `TechSupport`, `TotalCharges`, `MonthlyCharges`, `PaperlessBilling`, `DeviceProtection`, `Dependents`, `SeniorCitizen`, `Partner`, `OnlineBackup`.

---

## ⚙️ Key Technical Features

1. **Preprocessing:**
   - Cleans white-space noise in numerical fields like `TotalCharges`.
   - Uses `SimpleImputer` for handling missing numeric (mean) and categorical (mode) values.
   - Encodes categorical labels with `LabelEncoder`.

2. **Data Imbalance & Pipeline Architecture:**
   - Leverages `imblearn.pipeline.Pipeline` to prevent data leakage during feature transformations.
   - Applies **PolynomialFeatures** (degree 2 interaction terms) and **StandardScaler**.
   - Handles class imbalance cleanly using **SMOTE** oversampling on the training splits.

3. **Supervised ML Classifiers Evaluated:**
   - Logistic Regression
   - Random Forest Classifier
   - Decision Tree Classifier
   - K-Nearest Neighbors (KNN)

4. **Inference Readiness:**
   - Includes a standalone prediction function `predict_sample()` that safely reorders and encodes single unseen customer records before predicting.

---

## 📁 Repository Structure

```text
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Telco Customer Churn dataset
├── main.py                               # Complete ML training & prediction script
├── README.md                             # Project documentation
