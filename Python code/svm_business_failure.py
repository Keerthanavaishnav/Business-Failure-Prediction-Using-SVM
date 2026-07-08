# ============================================
# BUSINESS FAILURE PREDICTION USING SVM
# ============================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve
)

# ============================================
# 2. LOAD DATASET
# ============================================

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "svm dataset.csv")

# Read CSV (handles encoding issue)
data = pd.read_csv(file_path, encoding="latin1")

print("First 5 Rows")
print(data.head())

print("\nDataset Shape :", data.shape)

print("\nColumns")
print(data.columns)

# ============================================
# 3. REMOVE UNWANTED COLUMNS
# ============================================

data.drop(
    columns=["Training/Test", "Unnamed: 9", "Unnamed: 10"],
    errors="ignore",
    inplace=True
)

# ============================================
# 4. DATA CLEANING
# ============================================

print("\nMissing Values")
print(data.isnull().sum())

# Convert all feature columns to numeric
for col in data.columns:
    if col != "Business_Failure":
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove missing values
data.dropna(inplace=True)

print("\nDataset Shape After Cleaning :", data.shape)

# ============================================
# 5. FEATURE & TARGET SEPARATION
# ============================================

X = data.drop("Business_Failure", axis=1)

y = data["Business_Failure"]

# Convert target labels
# -1 = Survived -> 0
#  1 = Failed    -> 1
y = y.replace({-1: 0, 1: 1})

print("\nTarget Class Distribution")
print(y.value_counts())

# ============================================
# 6. TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Size :", len(X_train))
print("Testing Size  :", len(X_test))

# ============================================
# 7. FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 8. HYPERPARAMETER TUNING
# ============================================

param_grid = {
    'C': [0.1, 1, 5, 10],
    'gamma': [0.001, 0.01, 0.1, 'scale'],
    'kernel': ['rbf']
}

svm = SVC(
    probability=True,
    class_weight='balanced',
    random_state=42
)

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=2,
    error_score='raise'
)

grid_search.fit(X_train_scaled, y_train)

svm_best = grid_search.best_estimator_

print("\nBest Parameters")
print(grid_search.best_params_)

print("\nBest Cross Validation Score")
print(grid_search.best_score_)

print("\nSupport Vectors")
print(svm_best.n_support_)

# ============================================
# 9. CROSS VALIDATION
# ============================================

cv_scores = cross_val_score(
    svm_best,
    X_train_scaled,
    y_train,
    cv=5
)

print("\nCross Validation Scores")
print(cv_scores)

print("Mean CV Accuracy :", cv_scores.mean())

# ============================================
# 10. SUPPORT VECTORS
# ============================================

print("\nSupport Vectors Per Class")
print(svm_best.n_support_)

print("Total Support Vectors :", svm_best.support_vectors_.shape[0])

# ============================================
# 11. TRAINING PERFORMANCE
# ============================================

y_train_pred = svm_best.predict(X_train_scaled)

print("\n========== TRAINING ==========")

print("Accuracy :", accuracy_score(y_train, y_train_pred))

print(classification_report(y_train, y_train_pred))

# ============================================
# 12. TEST PERFORMANCE
# ============================================

y_test_pred = svm_best.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
balanced = balanced_accuracy_score(y_test, y_test_pred)
kappa = cohen_kappa_score(y_test, y_test_pred)
mcc = matthews_corrcoef(y_test, y_test_pred)

print("\n========== TEST ==========")

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)
print("Balanced Accuracy :", balanced)
print("Cohen Kappa :", kappa)
print("Matthews Correlation :", mcc)

print("\nClassification Report")
print(classification_report(y_test, y_test_pred))

# ============================================
# 13. ROC AUC
# ============================================

scores = svm_best.decision_function(X_test_scaled)

roc = roc_auc_score(y_test, scores)

print("\nROC AUC :", roc)

# ============================================
# 14. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_test_pred)

print("\nConfusion Matrix")
print(cm)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

print("Specificity :", specificity)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Survived","Failed"],
    yticklabels=["Survived","Failed"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ============================================
# 15. ROC CURVE
# ============================================

fpr, tpr, _ = roc_curve(y_test, scores)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"AUC = {roc:.3f}")

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

print("\nModel Completed Successfully")