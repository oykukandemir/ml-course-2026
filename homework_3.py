"""ML Zoomcamp 2026: lead-scoring classification homework."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(df_train, df_val, y_train, y_val, C=1.0):
    # Learn one-hot encoding from the training data only.
    train_dicts = df_train.to_dict(orient="records")
    val_dicts = df_val.to_dict(orient="records")

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    model = LogisticRegression(
        solver="liblinear",
        C=C,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)


# Read the pinned dataset. Use a local copy if it is next to this script.
url = (
    "https://raw.githubusercontent.com/DataTalksClub/"
    "machine-learning-zoomcamp/main/cohorts/2026/data/"
    "course_lead_scoring_2026.csv"
)
csv_path = Path(__file__).resolve().parent / "course_lead_scoring_2026.csv"
df = pd.read_csv(csv_path if csv_path.exists() else url)

categorical = [
    "lead_source",
    "industry",
    "employment_status",
    "location",
]
numerical = [
    "annual_income",
    "number_of_courses_viewed",
    "interaction_count",
    "lead_score",
]

print("Dataset shape:", df.shape)
print("\nMissing values before filling:")
print(df.isna().sum())

# Fill categorical and numerical features as required by the homework.
df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)


# Question 1: Find the most frequent industry after filling missing values.
industry_counts = df["industry"].value_counts()
print("\nQ1 - Industry counts:")
print(industry_counts)
print("Q1 - Most frequent industry:", industry_counts.idxmax())


# Question 2: Compare only the four pairs listed in the question.
correlations = df[numerical].corr()
print("\nQ2 - Correlation matrix:")
print(correlations)

pairs = [
    ("interaction_count", "lead_score"),
    ("number_of_courses_viewed", "lead_score"),
    ("number_of_courses_viewed", "interaction_count"),
    ("annual_income", "interaction_count"),
]
pair_scores = {}
for first, second in pairs:
    score = correlations.loc[first, second]
    pair_scores[(first, second)] = score
    print(f"Q2 - {first} and {second}: {score:.6f}")

best_pair = max(pair_scores, key=pair_scores.get)
print("Q2 - Biggest correlation:", " and ".join(best_pair))


# Split exactly as specified: 60% training, 20% validation, 20% test.
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42
)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Save the targets separately and remove them from the feature tables.
y_train = df_train["converted"].to_numpy()
y_val = df_val["converted"].to_numpy()
y_test = df_test["converted"].to_numpy()

df_train = df_train.drop(columns="converted")
df_val = df_val.drop(columns="converted")
df_test = df_test.drop(columns="converted")
df_full_train = df_full_train.drop(columns="converted")

print("\nTraining features:", df_train.shape)
print("Validation features:", df_val.shape)
print("Test features:", df_test.shape)


# Question 3: Calculate mutual information using training data only.
mi_scores = {}
print("\nQ3 - Mutual information:")
for column in categorical:
    score = mutual_info_score(y_train, df_train[column])
    mi_scores[column] = round(score, 2)
    print(f"{column}: {score:.6f} -> {mi_scores[column]:.2f}")

best_mi = max(mi_scores, key=mi_scores.get)
print("Q3 - Biggest rounded mutual information:", best_mi)


# Question 4: Train with all features and evaluate validation accuracy.
baseline_accuracy = train_and_evaluate(df_train, df_val, y_train, y_val)
print("\nQ4 - Validation accuracy:", baseline_accuracy)
print("Q4 - Rounded accuracy:", round(baseline_accuracy, 2))


# Question 5: Remove one original feature at a time and retrain.
# Keep the full precision of the baseline and reduced-model accuracies.
elimination_results = []
for column in df_train.columns:
    accuracy = train_and_evaluate(
        df_train.drop(columns=column),
        df_val.drop(columns=column),
        y_train,
        y_val,
    )
    difference = baseline_accuracy - accuracy
    elimination_results.append(
        {
            "removed_feature": column,
            "accuracy_without_feature": accuracy,
            "difference": difference,
        }
    )

elimination = pd.DataFrame(elimination_results).set_index("removed_feature")
print("\nQ5 - Feature elimination:")
print(elimination.to_string(float_format=lambda value: f"{value:.6f}"))

q5_options = ["lead_source", "number_of_courses_viewed", "interaction_count"]
least_useful = elimination.loc[q5_options, "difference"].idxmin()
print("Q5 - Smallest difference among the listed options:", least_useful)


# Question 6: Compare regularization settings using all features.
C_values = [0.000001, 0.00001, 0.0001, 0.001]
regularization_scores = {}
print("\nQ6 - Validation accuracy for each C:")
for C in C_values:
    accuracy = train_and_evaluate(df_train, df_val, y_train, y_val, C=C)
    regularization_scores[C] = round(accuracy, 3)
    print(f"C={C:g}: {accuracy:.6f} -> {regularization_scores[C]:.3f}")

best_accuracy = max(regularization_scores.values())
best_C_values = [
    C for C, accuracy in regularization_scores.items()
    if accuracy == best_accuracy
]
print("Q6 - Best C value(s) after rounding:", best_C_values)

# The test set is held aside; all requested model comparisons use validation.
