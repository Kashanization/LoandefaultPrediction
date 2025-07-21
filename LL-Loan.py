
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


df = pd.read_csv("C:/Users/pegahg/Desktop/LoanDefaultPrediction/loan_data_encoded.csv")


X = df.drop("Default", axis=1)
y = df["Default"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


model_logreg = LogisticRegression(
    class_weight="balanced",   
    solver='liblinear',        
    random_state=42
)


model_logreg.fit(X_train, y_train)


y_pred_prob = model_logreg.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_prob > 0.5).astype("int32")  # threshold 0.5 برای شروع


print("Classification Report:")
print(classification_report(y_test, y_pred_class))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

print("ROC AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))
