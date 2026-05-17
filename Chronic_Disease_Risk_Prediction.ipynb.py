!pip install -q pandas matplotlib scikit-learn

import pandas as pd
import numpy as np

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("diabetes.csv", low_memory=False)

# Replace impossible zeros with NaN (clinical logic)
cols_with_zero_issue = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero_issue] = df[cols_with_zero_issue].replace(0, np.nan)

# Fill with median (robust for clinical data)
df.fillna(df.median(), inplace=True)

print(df.isnull().sum())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report, confusion_matrix

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Sensitivity (Recall):", recall_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

feat_importance = pd.Series(rf.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.show()

risk_prob = rf.predict_proba(X_test)[:,1]

def risk_level(p):
    if p < 0.30:
        return "Low"
    elif p < 0.60:
        return "Medium"
    else:
        return "High"

risk_labels = [risk_level(p) for p in risk_prob]

from joblib import dump

dump(rf, "diabetes_rf_model.joblib")
