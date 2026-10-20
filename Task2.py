# TASK 2: PREPROCESSING, FEATURES AND PIPELINE
# Dataset: Synthetic Loan Approval Dataset

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression


# 1. CREATE A SAMPLE LOAN APPROVAL DATASET


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
    "application_date": pd.to_datetime(
        "2022-01-01"
    ) + pd.to_timedelta(
        rng.integers(0, 1460, n_samples), unit="D"
    )
})

# Create a target using a simple synthetic rule with noise
approval_score = (
    0.000025 * df["income"]
    - 0.000012 * df["loan_amount"]
    + 0.015 * df["age"]
    + rng.normal(0, 0.8, n_samples)
)

df["loan_approved"] = (
    approval_score > approval_score.median()
).astype(int)

# Introduce missing values
for column in ["income", "loan_amount", "age"]:
    missing_idx = rng.choice(n_samples, 25, replace=False)
    df.loc[missing_idx, column] = np.nan

for column in ["employment_type", "education"]:
    missing_idx = rng.choice(n_samples, 20, replace=False)
    df.loc[missing_idx, column] = np.nan

print("=" * 60)
print("1. DATASET")
print("=" * 60)
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())


# 2. FEATURE ENGINEERING
#    Create features inside a transformer so that feature
#    creation is part of the pipeline.

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Date decomposition: extract application year and month
        dates = pd.to_datetime(X["application_date"])
        X["application_year"] = dates.dt.year
        X["application_month"] = dates.dt.month

        # Interaction feature: income divided by loan amount
        X["income_loan_ratio"] = (
            X["income"] / X["loan_amount"].replace(0, np.nan)
        )

        # Log transform of skewed income
        X["log_income"] = np.log1p(X["income"].clip(lower=0))

        # Remove the original date column after extracting features
        X = X.drop(columns=["application_date"])

        return X


X = df.drop(columns=["loan_approved"])
y = df["loan_approved"]

# Inspect engineered features for demonstration
engineered = LoanFeatureEngineer().fit_transform(X)

print("\n" + "=" * 60)
print("2. ENGINEERED FEATURES")
print("=" * 60)
print(engineered[
    ["application_year", "application_month",
     "income_loan_ratio", "log_income"]
].head())

# 3. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))

# 4. PREPROCESSING PIPELINES

numeric_features = [
    "income",
    "loan_amount",
    "age",
    "application_year",
    "application_month",
    "income_loan_ratio",
    "log_income"
]

categorical_features = ["employment_type"]
ordinal_features = ["education"]

# Numeric: median imputation and scaling
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Nominal categories: most frequent imputation and one-hot encoding
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Ordered categories: most frequent imputation and ordinal encoding
ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[["High School", "Graduate", "Postgraduate"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
    ("ordinal", ordinal_pipeline, ordinal_features)
])


# 5. COMPLETE PIPELINE

model_pipeline = Pipeline([
    ("feature_engineering", LoanFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

# 6. FIVE-FOLD CROSS-VALIDATION

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\n" + "=" * 60)
print("6. CROSS-VALIDATION")
print("=" * 60)
print("Fold accuracies:", np.round(cv_scores, 4))
print(f"Pipeline CV accuracy: {cv_scores.mean():.2f}")
print(f"Standard deviation: {cv_scores.std():.2f}")

# 7. FIT THE PIPELINE

model_pipeline.fit(X_train, y_train)

test_accuracy = model_pipeline.score(X_test, y_test)

print("\n" + "=" * 60)
print("7. TEST EVALUATION")
print("=" * 60)
print(f"Test accuracy: {test_accuracy:.2f}")


# 8. TRY KNN IMPUTATION ON NUMERIC COLUMNS
#    Demonstration on training data only.

print("\n" + "=" * 60)
print("8. KNN IMPUTATION")
print("=" * 60)

knn_imputer = KNNImputer(n_neighbors=5)

X_train_numeric = X_train[
    ["income", "loan_amount", "age"]
]

X_train_knn = knn_imputer.fit_transform(X_train_numeric)

print("Numeric data shape before KNN imputation:",
      X_train_numeric.shape)
print("Numeric data shape after KNN imputation:",
      X_train_knn.shape)
print("Missing values after KNN imputation:",
      np.isnan(X_train_knn).sum())


# 9. SAVE THE FITTED PIPELINE

joblib.dump(model_pipeline, "loan_approval_pipeline.joblib")

print("\nFitted pipeline saved as loan_approval_pipeline.joblib")



# 10. LOAD THE PIPELINE AND PREDICT 3 NEW RAW ROWS

loaded_pipeline = joblib.load("loan_approval_pipeline.joblib")

new_rows = pd.DataFrame({
    "income": [65000.0, 120000.0, np.nan],
    "loan_amount": [20000.0, 30000.0, 15000.0],
    "age": [30.0, 45.0, 28.0],
    "employment_type": [
        "Salaried", "Self-employed", np.nan
    ],
    "education": [
        "Graduate", "Postgraduate", "High School"
    ],
    "application_date": pd.to_datetime([
        "2025-03-15", "2025-06-20", "2025-09-10"
    ])
})

predictions = loaded_pipeline.predict(new_rows)

print("\n" + "=" * 60)
print("10. PREDICTIONS FOR 3 NEW RAW ROWS")
print("=" * 60)
print("Predictions:", predictions)
print("0 = Not approved, 1 = Approved")