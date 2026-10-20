# ============================================================
# TASK 1: CONCEPTS, SPLITS AND LEAKAGE
# Dataset: Wine Dataset (Scikit-learn)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

data = load_wine()

X = data.data
y = data.target

print("=" * 60)
print("1. WINE DATASET")
print("=" * 60)

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Number of samples:", X.shape[0])
print("Wine classes:", data.target_names)


# ============================================================
# 2. SPLIT DATA: 60% TRAIN, 20% VALIDATION, 20% TEST
# ============================================================

# First, reserve 20% for testing
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split the remaining 80% into 60% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\n" + "=" * 60)
print("2. TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

print(f"Training samples:   {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test samples:       {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print("\nTest set is kept aside until final evaluation.")


# ============================================================
# 3. DEMONSTRATE OVERFITTING USING DECISION TREES
# ============================================================

print("\n" + "=" * 60)
print("3. DECISION TREE: OVERFITTING")
print("=" * 60)

depths = [1, 3, 5, 10, None]

train_scores = []
val_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    # Train using training data only
    model.fit(X_train, y_train)

    # Calculate training accuracy
    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    # Calculate validation accuracy
    val_accuracy = accuracy_score(
        y_val,
        model.predict(X_val)
    )

    train_scores.append(train_accuracy)
    val_scores.append(val_accuracy)

    print(
        f"Depth {str(depth):>4}: "
        f"train {train_accuracy:.2f} "
        f"val {val_accuracy:.2f}"
    )

# Plot training vs validation accuracy
depth_labels = [str(depth) for depth in depths]

plt.figure(figsize=(9, 5))

plt.plot(
    depth_labels,
    train_scores,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    depth_labels,
    val_scores,
    marker="s",
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Training vs Validation Accuracy")
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("overfitting_plot.png", dpi=300)
plt.show()


# ============================================================
# 4. LOGISTIC REGRESSION: 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("4. LOGISTIC REGRESSION: 5-FOLD CROSS-VALIDATION")
print("=" * 60)

# Scaling is fitted separately within each training fold
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    logistic_model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("Fold accuracies:", np.round(cv_scores, 4))
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")
print(f"CV: {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")


# ============================================================
# 5. DEMONSTRATE DATA LEAKAGE
# ============================================================

print("\n" + "=" * 60)
print("5. DATA LEAKAGE: INCORRECT VS CORRECT SCALING")
print("=" * 60)

# INCORRECT METHOD:
# Fit the scaler using the entire dataset before splitting
scaler_wrong = StandardScaler()
X_scaled_wrong = scaler_wrong.fit_transform(X)

print("\nIncorrect scaling:")
print("Scaler fitted using the entire dataset.")
print("First 3 feature means:", scaler_wrong.mean_[:3])

# CORRECT METHOD:
# Fit the scaler using training data only
scaler_correct = StandardScaler()

X_train_scaled = scaler_correct.fit_transform(X_train)

# Transform validation and test data using the fitted scaler
X_val_scaled = scaler_correct.transform(X_val)
X_test_scaled = scaler_correct.transform(X_test)

print("\nCorrect scaling:")
print("Scaler fitted using training data only.")
print("First 3 training feature means:", scaler_correct.mean_[:3])

print("\nScaled data shapes:")
print("Training:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Test:", X_test_scaled.shape)

print("\nData leakage explanation:")
print("1. Scaling before splitting uses information from the entire dataset.")
print("2. Validation and test data influence preprocessing statistics.")
print("3. Correct scaling fits StandardScaler on training data only.")
print("4. The fitted scaler transforms validation and test data without refitting.")


# ============================================================
# 6. SUPERVISED VS UNSUPERVISED LEARNING
# ============================================================

print("\n" + "=" * 60)
print("6. SUPERVISED VS UNSUPERVISED LEARNING")
print("=" * 60)

problems = [
    ("Spam detection", "Supervised",
     "Uses labelled spam and non-spam emails."),

    ("Customer segmentation", "Unsupervised",
     "Groups customers based on similarities without predefined labels."),

    ("House price prediction", "Supervised",
     "Learns from historical houses with known prices."),

    ("Anomaly detection", "Unsupervised (typical)",
     "Identifies unusual patterns without labelled anomalies.")
]

for problem, learning_type, reason in problems:
    print(f"\nProblem: {problem}")
    print(f"Type: {learning_type}")
    print(f"Reason: {reason}")


# ============================================================
# 7. MACHINE LEARNING WORKFLOW IN 7 STEPS
# ============================================================

print("\n" + "=" * 60)
print("7. MACHINE LEARNING WORKFLOW")
print("=" * 60)

workflow = [
    "1. Data collection: Gather relevant data.",
    "2. Data exploration and cleaning: Handle missing values, duplicates, and outliers.",
    "3. Data splitting: Divide data into training, validation, and test sets.",
    "4. Feature engineering and preprocessing: Prepare features and scale without leakage.",
    "5. Model training: Train suitable machine learning algorithms.",
    "6. Validation and model selection: Compare models, tune parameters, and check overfitting.",
    "7. Final testing and deployment: Evaluate on held-out test data, deploy, and monitor."
]

for step in workflow:
    print(step)


# ============================================================
# 8. FINAL TEST EVALUATION
# Run only after completing model selection.
# ============================================================

print("\n" + "=" * 60)
print("8. FINAL TEST-SET EVALUATION")
print("=" * 60)

# Refit the selected Logistic Regression pipeline using
# training + validation data after model selection.
X_train_final = np.vstack((X_train, X_val))
y_train_final = np.concatenate((y_train, y_val))

final_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)

final_model.fit(X_train_final, y_train_final)

# Evaluate once on the untouched test set
y_test_pred = final_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Final test accuracy: {test_accuracy:.4f}")
print(f"Final test accuracy (%): {test_accuracy * 100:.2f}%")

print("\nAssessment completed successfully.")
print("Overfitting plot saved as overfitting_plot.png")