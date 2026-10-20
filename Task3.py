
# ============================================================
# TASK 3: MODELS - SUPERVISED AND UNSUPERVISED
# Dataset: Synthetic Loan Approval Dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from xgboost import XGBClassifier


# ============================================================
# 1. CREATE DATASET
# ============================================================

rng = np.random.default_rng(42)
n_samples = 500

df = pd.DataFrame({
    "income": rng.integers(20000, 150001, n_samples).astype(float),
    "loan_amount": rng.integers(5000, 60001, n_samples).astype(float),
    "age": rng.integers(21, 66, n_samples).astype(float),
    "employment_type": rng.choice(
        ["Salaried", "Self-employed", "Unemployed"], n_samples
    ),
    "education": rng.choice(
        ["High School", "Graduate", "Postgraduate"], n_samples
    ),
    "application_date": pd.to_datetime("2022-01-01")
    + pd.to_timedelta(rng.integers(0, 1460, n_samples), unit="D")
})

approval_score = (
    0.000025 * df["income"]
    - 0.000012 * df["loan_amount"]
    + 0.015 * df["age"]
    + rng.normal(0, 0.8, n_samples)
)

df["loan_approved"] = (
    approval_score > approval_score.median()
).astype(int)

for column in ["income", "loan_amount", "age"]:
    missing_idx = rng.choice(n_samples, 25, replace=False)
    df.loc[missing_idx, column] = np.nan

for column in ["employment_type", "education"]:
    missing_idx = rng.choice(n_samples, 20, replace=False)
    df.loc[missing_idx, column] = np.nan

print("=" * 65)
print("TASK 3: SUPERVISED AND UNSUPERVISED MODELS")
print("=" * 65)
print(df.head())


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        dates = pd.to_datetime(X["application_date"])
        X["application_year"] = dates.dt.year
        X["application_month"] = dates.dt.month

        X["income_loan_ratio"] = (
            X["income"] / X["loan_amount"].replace(0, np.nan)
        )

        X["log_income"] = np.log1p(X["income"].clip(lower=0))
        return X.drop(columns=["application_date"])


X = df.drop(columns=["loan_approved"])
y = df["loan_approved"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

numeric_features = [
    "income", "loan_amount", "age", "application_year",
    "application_month", "income_loan_ratio", "log_income"
]

categorical_features = ["employment_type"]
ordinal_features = ["education"]


def create_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["High School", "Graduate", "Postgraduate"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("ordinal", ordinal_pipeline, ordinal_features)
    ])


def create_pipeline(classifier):
    return Pipeline([
        ("features", LoanFeatureEngineer()),
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])


# ============================================================
# 3. FOUR CLASSIFIERS
# ============================================================

print("\n" + "=" * 65)
print("CLASSIFICATION MODELS")
print("=" * 65)

logistic_model = create_pipeline(
    LogisticRegression(max_iter=2000)
)

tree_model = create_pipeline(
    DecisionTreeClassifier(max_depth=5, random_state=42)
)

forest_model = create_pipeline(
    RandomForestClassifier(
        n_estimators=200,
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
)

logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

# XGBoost: transform training and validation data using a
# preprocessor fitted on training data only.
feature_engineer = LoanFeatureEngineer()
X_train_features = feature_engineer.fit_transform(X_train)
X_val_features = feature_engineer.transform(X_val)

xgb_preprocessor = create_preprocessor()
X_train_transformed = xgb_preprocessor.fit_transform(
    X_train_features, y_train
)
X_val_transformed = xgb_preprocessor.transform(X_val_features)

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_transformed,
    y_train,
    eval_set=[(X_val_transformed, y_val)],
    verbose=False
)

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model
}

results = []

for name, model in models.items():
    predictions = model.predict(X_val)
    score = accuracy_score(y_val, predictions)
    results.append({"Model": name, "Validation score": score})
    print(f"{name}: Validation accuracy = {score:.4f}")

xgb_predictions = xgb_model.predict(X_val_transformed)
xgb_score = accuracy_score(y_val, xgb_predictions)

results.append({"Model": "XGBoost", "Validation score": xgb_score})
print(f"XGBoost: Validation accuracy = {xgb_score:.4f}")

print("\nRandom Forest OOB score:")
print(f"{forest_model.named_steps['classifier'].oob_score_:.4f}")

print("XGBoost best iteration:", xgb_model.best_iteration)


# ============================================================
# 4. LOGISTIC REGRESSION: PROBABILITIES VS PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("LOGISTIC REGRESSION: PREDICT_PROBA VS PREDICT")
print("=" * 65)

first_five = X_val.iloc[:5]

probabilities = logistic_model.predict_proba(first_five)
predictions = logistic_model.predict(first_five)

for i in range(len(first_five)):
    print(f"\nRow {i + 1}")
    print("Probabilities [class 0, class 1]:",
          np.round(probabilities[i], 4))
    print("predict():", predictions[i])


# ============================================================
# 5. TOP 5 RANDOM FOREST FEATURES
# ============================================================

print("\n" + "=" * 65)
print("TOP 5 RANDOM FOREST FEATURES")
print("=" * 65)

rf_classifier = forest_model.named_steps["classifier"]
rf_feature_names = (
    forest_model.named_steps["preprocessor"].get_feature_names_out()
)

rf_importance = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": rf_classifier.feature_importances_
}).sort_values("Importance", ascending=False)

print(rf_importance.head(5).to_string(index=False))


# ============================================================
# 6. TOP 5 XGBOOST FEATURES
# ============================================================

print("\n" + "=" * 65)
print("TOP 5 XGBOOST FEATURES")
print("=" * 65)

xgb_importance = pd.DataFrame({
    "Feature": xgb_preprocessor.get_feature_names_out(),
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)

print(xgb_importance.head(5).to_string(index=False))


# ============================================================
# 7. REGRESSION: PREDICT LOAN AMOUNT
# ============================================================

print("\n" + "=" * 65)
print("LINEAR REGRESSION, RIDGE AND LASSO")
print("=" * 65)

# Remove rows where the target loan amount is missing.
regression_df = df.dropna(subset=["loan_amount"]).copy()

X_reg = regression_df.drop(
    columns=["loan_approved", "loan_amount"]
)
y_reg = regression_df["loan_amount"]

X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)


class RegressionFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dates = pd.to_datetime(X["application_date"])
        X["application_year"] = dates.dt.year
        X["application_month"] = dates.dt.month
        X["log_income"] = np.log1p(X["income"].clip(lower=0))
        return X.drop(columns=["application_date"])


reg_numeric = [
    "income", "age", "application_year",
    "application_month", "log_income"
]

reg_categorical = ["employment_type"]
reg_ordinal = ["education"]


def create_regression_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), reg_numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), reg_categorical),
        ("ordinal", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[
                    ["High School", "Graduate", "Postgraduate"]
                ],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ]), reg_ordinal)
    ])


regressors = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=10.0),
    "Lasso": Lasso(alpha=100.0, max_iter=10000)
}

coefficient_results = {}

for name, regressor in regressors.items():
    reg_pipeline = Pipeline([
        ("features", RegressionFeatureEngineer()),
        ("preprocessor", create_regression_preprocessor()),
        ("regressor", regressor)
    ])

    reg_pipeline.fit(X_reg_train, y_reg_train)
    score = reg_pipeline.score(X_reg_val, y_reg_val)

    fitted_regressor = reg_pipeline.named_steps["regressor"]
    feature_names = (
        reg_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )

    coefficient_results[name] = pd.Series(
        fitted_regressor.coef_, index=feature_names
    )

    print(f"\n{name}")
    print(f"Validation R-squared: {score:.4f}")
    print("First 10 coefficients:")
    print(coefficient_results[name].head(10).round(4))

lasso_coefficients = coefficient_results["Lasso"]
zero_count = np.sum(np.isclose(lasso_coefficients.values, 0))

print("\nNumber of zero Lasso coefficients:", int(zero_count))

print("\nCoefficient comparison:")
print(pd.DataFrame(coefficient_results).round(4).to_string())


# ============================================================
# 8. K-MEANS: K = 2 TO 8
# ============================================================

print("\n" + "=" * 65)
print("K-MEANS CLUSTERING")
print("=" * 65)

cluster_features = ["income", "loan_amount", "age"]

cluster_data = df[cluster_features]

cluster_imputer = SimpleImputer(strategy="median")
cluster_scaler = StandardScaler()

cluster_imputed = cluster_imputer.fit_transform(cluster_data)
cluster_scaled = cluster_scaler.fit_transform(cluster_imputed)

k_values = list(range(2, 9))
inertias = []
silhouette_scores = []
cluster_models = {}

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(cluster_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(cluster_scaled, labels)

    inertias.append(inertia)
    silhouette_scores.append(silhouette)
    cluster_models[k] = (kmeans, labels)

    print(
        f"k={k}: Inertia={inertia:.2f}, "
        f"Silhouette={silhouette:.4f}"
    )


# ============================================================
# 9. PLOT ELBOW CURVE AND SILHOUETTE SCORES
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(k_values, inertias, marker="o")
axes[0].set_title("K-Means Elbow Curve")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_xticks(k_values)
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_values, silhouette_scores, marker="s", color="orange")
axes[1].set_title("Silhouette Scores")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(k_values)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task3_elbow_silhouette.png", dpi=300)
plt.show()


# ============================================================
# 10. DESCRIBE CLUSTERS
# ============================================================

best_k = k_values[int(np.argmax(silhouette_scores))]
best_kmeans, best_labels = cluster_models[best_k]

cluster_summary_data = df[cluster_features].copy()
cluster_summary_data["Cluster"] = best_labels

cluster_summary = cluster_summary_data.groupby("Cluster").agg(
    Count=("income", "size"),
    Average_Income=("income", "mean"),
    Average_Loan_Amount=("loan_amount", "mean"),
    Average_Age=("age", "mean")
).round(2)

print("\n" + "=" * 65)
print(f"CLUSTER DESCRIPTIONS: k={best_k}")
print("=" * 65)
print(cluster_summary)

print("\nInterpretation:")
print("Compare average income, loan amount and age across clusters.")
print("Clusters are discovered from similarities, not predefined labels.")


# ============================================================
# 11. RESULTS TABLE
# ============================================================

print("\n" + "=" * 65)
print("MODEL COMPARISON TABLE")
print("=" * 65)

results_table = pd.DataFrame(results).sort_values(
    "Validation score", ascending=False
)

print(results_table.to_string(index=False))
results_table.to_csv("task3_model_results.csv", index=False)

print("\nFiles saved:")
print("1. task3_elbow_silhouette.png")
print("2. task3_model_results.csv")
