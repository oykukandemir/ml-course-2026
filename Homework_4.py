"""ML Zoomcamp 2026 - Homework 4: evaluating a classifier."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split


def train_and_predict(df_train, df_val, C=1.0):
    # Keep the target separate from the features in both datasets.
    y_train = df_train["converted"].to_numpy()
    train_features = df_train.drop(columns=["converted"])
    val_features = df_val.drop(columns=["converted"])

    # Learn the encoding from this training set only.
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_features.to_dict(orient="records"))
    X_val = dv.transform(val_features.to_dict(orient="records"))

    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
    model.fit(X_train, y_train)

    # AUC and threshold comparisons need probabilities for class 1.
    return model.predict_proba(X_val)[:, 1]


def cross_validate(df_full_train, C=1.0):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_fold_train = df_full_train.iloc[train_idx]
        df_fold_val = df_full_train.iloc[val_idx]

        y_val = df_fold_val["converted"].to_numpy()
        y_probability = train_and_predict(df_fold_train, df_fold_val, C=C)
        scores.append(roc_auc_score(y_val, y_probability))

    return np.array(scores)


# Read the pinned dataset. Prefer a local CSV next to this script.
output_dir = Path(__file__).resolve().parent
csv_path = output_dir / "course_lead_scoring_2026.csv"
url = (
    "https://raw.githubusercontent.com/DataTalksClub/"
    "machine-learning-zoomcamp/main/cohorts/2026/data/"
    "course_lead_scoring_2026.csv"
)
df = pd.read_csv(csv_path if csv_path.exists() else url)

categorical = ["lead_source", "industry", "employment_status", "location"]
numerical = [
    "annual_income",
    "number_of_courses_viewed",
    "interaction_count",
    "lead_score",
]

print("Dataset shape:", df.shape)
print("\nMissing values before filling:")
print(df.isna().sum())

df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)

# Use the two exact split calls required by the homework.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

y_train = df_train["converted"].to_numpy()
y_val = df_val["converted"].to_numpy()

print("\nTraining:", df_train.shape)
print("Validation:", df_val.shape)
print("Test:", df_test.shape)


# Q1: Use each numerical feature directly as a score on the training set.
feature_auc = {}
print("\nQ1 - Numerical feature AUC:")
for feature in numerical:
    values = df_train[feature].to_numpy()
    auc = roc_auc_score(y_train, values)
    inverted = auc < 0.5

    if inverted:
        auc = roc_auc_score(y_train, -values)

    feature_auc[feature] = auc
    print(f"{feature}: AUC={auc:.6f}, inverted={inverted}")

best_feature = max(feature_auc, key=feature_auc.get)
print("Q1 answer:", best_feature)


# Q2: Train on all features and evaluate validation ROC AUC.
y_probability = train_and_predict(df_train, df_val, C=1.0)
validation_auc = roc_auc_score(y_val, y_probability)
print("\nQ2 - Validation AUC:", validation_auc)
print("Q2 answer:", round(validation_auc, 3))


# Q3 and Q4: Evaluate all 101 thresholds, including 0.00 and 1.00.
threshold_results = []
for threshold in np.arange(0.0, 1.01, 0.01):
    y_pred = y_probability >= threshold

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    threshold_results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "absolute_difference": abs(precision - recall),
        }
    )

metrics = pd.DataFrame(threshold_results)

# Ignore thresholds where both precision and recall are zero.
valid_metrics = metrics[
    (metrics["precision"] != 0) | (metrics["recall"] != 0)
]
intersection = valid_metrics.loc[valid_metrics["absolute_difference"].idxmin()]
best_f1 = metrics.loc[metrics["f1"].idxmax()]

print("\nQ3 - Closest precision and recall:")
print(intersection.to_string())
print("Q3 answer:", round(intersection["threshold"], 2))

print("\nQ4 - Maximum F1:")
print(best_f1.to_string())
print("Q4 answer:", round(best_f1["threshold"], 2))

metrics.to_csv(output_dir / "homework_04_threshold_metrics.csv", index=False)

# Plot the metrics against the decision threshold.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(metrics["threshold"], metrics["precision"], label="Precision")
axes[0].plot(metrics["threshold"], metrics["recall"], label="Recall")
axes[0].axvline(
    intersection["threshold"],
    color="black",
    linestyle="--",
    label=f"Closest match: {intersection['threshold']:.2f}",
)
axes[0].set_title("Precision and recall by threshold")
axes[0].set_ylabel("Score")

axes[1].plot(metrics["threshold"], metrics["f1"], label="F1", color="tab:green")
axes[1].axvline(
    best_f1["threshold"],
    color="black",
    linestyle="--",
    label=f"Best threshold: {best_f1['threshold']:.2f}",
)
axes[1].set_title("F1 by threshold")
axes[1].set_ylabel("F1 score")

for ax in axes:
    ax.set_xlabel("Decision threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

fig.tight_layout()
plot_path = output_dir / "homework_04_metrics.png"
fig.savefig(plot_path, dpi=180)
plt.close(fig)


# Q5: Five-fold cross-validation on the full training set.
scores_C1 = cross_validate(df_full_train, C=1.0)
print("\nQ5 - Fold AUC scores:", scores_C1)
print("Q5 - Mean AUC:", np.mean(scores_C1))
print("Q5 - AUC standard deviation:", np.std(scores_C1))
print("Q5 answer:", round(float(np.std(scores_C1)), 3))


# Q6: Use the same five folds for every C value.
C_values = [0.000001, 0.001, 1]
cv_results = []
fold_results = []
print("\nQ6 - Cross-validation results:")

for C in C_values:
    # Reuse Q5's scores when C=1 to avoid training the same models twice.
    scores = scores_C1 if C == 1 else cross_validate(df_full_train, C=C)
    mean_auc = float(np.mean(scores))
    std_auc = float(np.std(scores))

    cv_results.append(
        {
            "C": C,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "rounded_mean": round(mean_auc, 3),
            "rounded_std": round(std_auc, 3),
        }
    )

    for fold, auc in enumerate(scores, start=1):
        fold_results.append({"C": C, "fold": fold, "auc": auc})

    print(f"C={C:g}: mean AUC={mean_auc:.3f}, std={std_auc:.3f}")

# Compare rounded means, then rounded standard deviations, then C values.
best_result = min(
    cv_results,
    key=lambda result: (
        -result["rounded_mean"],
        result["rounded_std"],
        result["C"],
    ),
)
print("Q6 answer:", best_result["C"])

pd.DataFrame(cv_results).to_csv(
    output_dir / "homework_04_cross_validation.csv", index=False
)
pd.DataFrame(fold_results).to_csv(
    output_dir / "homework_04_fold_scores.csv", index=False
)

print("\nSaved plot:", plot_path)
# Keep the test set untouched during model and threshold selection.
