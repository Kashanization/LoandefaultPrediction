#  Loan Default Prediction

A machine learning project to predict whether a customer will default on a loan using structured financial data.  
The project explores and compares three ML models:

- ✅ Logistic Regression (baseline)
- ✅ Neural Network (with TensorFlow/Keras)
- ✅ XGBoost (tree-based ensemble)

---

##  Problem Statement

Given a dataset of loan applicants with personal and financial attributes, the goal is to predict the likelihood of defaulting on a personal loan.  
This is a **binary classification** problem with **imbalanced classes** (~11% default).

---

##  Dataset

- **Source**: Kaggle  
- **File**: `loan_data_encoded.csv` (preprocessed version included)  
- **Target**: `Default` (0 = not default, 1 = default)  
- **Size**: ~255,000 rows × ~20 features  
- **Type**: Structured tabular data

---

## Features (sample)

- `Income`, `CreditScore`, `EmploymentLength`, `LoanAmount`, `InterestRate`
- `NumCreditLines`, `DTIRatio`, `State`, `Purpose`, etc.
- All categorical variables were one-hot encoded for model use.

---

## Models & Evaluation

| Model              | Accuracy | Precision (1) | Recall (1) | F1 Score (1) | ROC AUC |
|--------------------|----------|---------------|------------|--------------|---------|
| Logistic Regression| 0.68     | 0.22          | 0.70       | 0.33         | 0.753   |
| Neural Network     | 0.87     | 0.22          | 0.74       | 0.34         | 0.757   |
| XGBoost            | 0.70     | 0.23          | 0.68       | 0.34         | 0.758   |

- **Threshold** tuning was performed (0.3–0.5)
- **Class imbalance** handled with class weights
- Models were evaluated using accuracy, recall, f1, and AUC metrics

---

##  File Guide

| File / Folder           | Description |
|-------------------------|-------------|
| `data/`                 | Preprocessed CSV dataset |
| `notebooks/`            | Jupyter notebooks for each ML model |
| `models/`               | Trained models (optional) |
| `README.md`             | Project documentation |
| `requirements.txt`      | Python libraries used |

---

##  Tech Stack

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Jupyter Notebooks

---

##  Future Work

- Hyperparameter tuning
- SMOTE or advanced resampling for class imbalance
- Deployment using Streamlit or Flask
- Model interpretability with SHAP or LIME

---

**Pegah Kashani**  
Machine Learning Engineer | Winnipeg, Canada  
📧 Kashani.pg@gmail.com
# LoandefaultPrediction
