# 📦 کتابخانه‌ها
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# 🧾 1. داده‌خوانی
df = pd.read_csv("C:/Users/pegahg/Desktop/LoanDefaultPrediction/loan_data_encoded.csv")

# 🎯 2. جدا کردن هدف (y) و ویژگی‌ها (X)
X = df.drop("Default", axis=1)
y = df["Default"]

# 📏 3. استانداردسازی
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✂️ 4. تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ⚖️ 5. تنظیم وزن برای class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 🌲 6. ساخت مدل XGBoost
model_xgb = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    random_state=42
)

# 🏋️‍♂️ 7. آموزش مدل
model_xgb.fit(X_train, y_train)

# 🔮 8. پیش‌بینی روی داده تست
y_pred_prob = model_xgb.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_prob > 0.5).astype("int32")  # آستانه تصمیم

# 📊 9. ارزیابی مدل
print("Classification Report:")
print(classification_report(y_test, y_pred_class))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

print("ROC AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))
