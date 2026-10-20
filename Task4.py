# ==========================================
# TASK 4: EVALUATION AND TUNING
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# PART 1: LOAD CLASSIFICATION DATASET
# ------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset:", data.DESCR.splitlines()[0])
print("Features:", X.shape)
print("Class distribution:")
print(dict(zip(*np.unique(y, return_counts=True))))

# 0 = malignant, 1 = benign in this dataset

# Keep the test set untouched until final evaluation.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split training data into fitting and validation sets.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)

# ------------------------------------------
# PART 2: DEFINE CLASSIFIERS
# ------------------------------------------

logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)

rf_model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)

# ------------------------------------------
# PART 3: GRID SEARCH FOR RANDOM FOREST
# Tune exactly 3 parameters
# ------------------------------------------

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

rf_search.fit(X_fit, y_fit)

best_rf = rf_search.best_estimator_

print("\n===== RANDOM FOREST GRID SEARCH =====")
print("Best parameters:", rf_search.best_params_)
print("Best CV ROC-AUC:", round(rf_search.best_score_, 4))

# ------------------------------------------
# PART 4: RANDOM SEARCH FOR XGBOOST
# n_iter = 15
# ------------------------------------------

xgb_params = {
    "n_estimators": [100, 150, 200, 300, 400],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=15,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_fit, y_fit)

best_xgb = xgb_search.best_estimator_

print("\n===== XGBOOST RANDOM SEARCH =====")
print("Best parameters:", xgb_search.best_params_)
print("Best CV ROC-AUC:", round(xgb_search.best_score_, 4))

# ------------------------------------------
# PART 5: TRAIN ALL THREE CLASSIFIERS
# ------------------------------------------

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

validation_results = {}

for name, model in models.items():

    model.fit(X_fit, y_fit)

    # Positive-class probabilities
    y_prob = model.predict_proba(X_val)[:, 1]

    # Default decision threshold = 0.50
    y_pred = (y_prob >= 0.50).astype(int)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)

    validation_results[name] = {
        "model": model,
        "probabilities": y_prob,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }

    print(f"\n===== {name} =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:\n", cm)

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        target_names=data.target_names,
        zero_division=0
    ))

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=data.target_names,
        yticklabels=data.target_names
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

# ------------------------------------------
# PART 6: ROC CURVES OF ALL 3 MODELS
# ------------------------------------------

plt.figure(figsize=(9, 6))

for name, result in validation_results.items():

    fpr, tpr, thresholds = roc_curve(
        y_val,
        result["probabilities"]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {result['auc']:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Model Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------
# PART 7: CHOOSE BEST MODEL
# Use validation ROC-AUC for model selection
# ------------------------------------------

best_name = max(
    validation_results,
    key=lambda name: validation_results[name]["auc"]
)

best_result = validation_results[best_name]
best_model = best_result["model"]

print("\nBest model on validation set:", best_name)
print("Validation ROC-AUC:", round(best_result["auc"], 4))
print("Validation F1:", round(best_result["f1"], 4))

# ------------------------------------------
# PART 8: FIND THRESHOLD FOR AT LEAST 75%
# RECALL USING VALIDATION DATA
# ------------------------------------------

y_prob_best = best_result["probabilities"]

threshold_options = np.linspace(0.01, 0.99, 99)
eligible_thresholds = []

for threshold in threshold_options:

    predictions = (y_prob_best >= threshold).astype(int)

    threshold_recall = recall_score(
        y_val, predictions, zero_division=0
    )

    threshold_precision = precision_score(
        y_val, predictions, zero_division=0
    )

    if threshold_recall >= 0.75:
        eligible_thresholds.append(
            (threshold, threshold_precision, threshold_recall)
        )

if eligible_thresholds:

    # Among thresholds achieving >= 75% recall,
    # select the one with the highest precision.
    chosen_threshold, chosen_precision, chosen_recall = max(
        eligible_thresholds,
        key=lambda item: (item[1], item[0])
    )

    print("\n===== THRESHOLD TUNING =====")
    print("Chosen threshold:", round(chosen_threshold, 4))
    print("Validation precision:", round(chosen_precision, 4))
    print("Validation recall:", round(chosen_recall, 4))

else:
    # Threshold near zero predicts nearly every sample
    # as positive; it should normally achieve high recall.
    chosen_threshold = 0.0

    threshold_predictions = (
        y_prob_best >= chosen_threshold
    ).astype(int)

    chosen_precision = precision_score(
        y_val, threshold_predictions, zero_division=0
    )
    chosen_recall = recall_score(
        y_val, threshold_predictions, zero_division=0
    )

    print("Using threshold 0.0")
    print("Validation precision:", chosen_precision)
    print("Validation recall:", chosen_recall)

# ------------------------------------------
# PART 9: FINAL EVALUATION ON TEST SET
# Evaluate the selected model only once
# ------------------------------------------

# Refit the selected model on all training data.
best_model.fit(X_train_full, y_train_full)

# Make test predictions once.
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

print("\n===== FINAL TEST SET RESULTS =====")
print("Best model:", best_name)
print("Threshold:", round(chosen_threshold, 4))
print("Test Accuracy:", round(
    accuracy_score(y_test, test_predictions), 4
))
print("Test Precision:", round(
    precision_score(y_test, test_predictions, zero_division=0), 4
))
print("Test Recall:", round(
    recall_score(y_test, test_predictions, zero_division=0), 4
))
print("Test F1:", round(
    f1_score(y_test, test_predictions, zero_division=0), 4
))
print("Test ROC-AUC:", round(
    roc_auc_score(y_test, test_probabilities), 4
))
print("Test Confusion Matrix:\n",
      confusion_matrix(y_test, test_predictions))

# ------------------------------------------
# PART 10: REGRESSION EVALUATION
# ------------------------------------------

regression_data = load_diabetes()

X_reg = regression_data.data
y_reg = regression_data.target

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)

from sklearn.ensemble import RandomForestRegressor

reg_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

reg_model.fit(X_reg_train, y_reg_train)

reg_predictions = reg_model.predict(X_reg_test)

# Mean baseline: predict the training-target mean
mean_value = np.mean(y_reg_train)
baseline_predictions = np.full(
    len(y_reg_test), mean_value
)

# Regression model metrics
model_mae = mean_absolute_error(y_reg_test, reg_predictions)
model_rmse = np.sqrt(
    mean_squared_error(y_reg_test, reg_predictions)
)
model_r2 = r2_score(y_reg_test, reg_predictions)

# Baseline metrics
baseline_mae = mean_absolute_error(
    y_reg_test, baseline_predictions
)
baseline_rmse = np.sqrt(
    mean_squared_error(y_reg_test, baseline_predictions)
)
baseline_r2 = r2_score(
    y_reg_test, baseline_predictions
)

print("\n===== REGRESSION VS MEAN BASELINE =====")

print("\nRandom Forest Regressor:")
print(f"MAE : {model_mae:.4f}")
print(f"RMSE: {model_rmse:.4f}")
print(f"R²  : {model_r2:.4f}")

print("\nMean Baseline:")
print(f"MAE : {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"R²  : {baseline_r2:.4f}")