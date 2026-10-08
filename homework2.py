import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Functions from the lessons
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


def train_linear_regression_reg(X, y, r=0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


def rmse(y, y_pred):
    error = y - y_pred
    return np.sqrt((error ** 2).mean())


# Load the pinned 2026 dataset and select the required columns.
url = (
    "https://raw.githubusercontent.com/DataTalksClub/"
    "machine-learning-zoomcamp/main/cohorts/2026/data/"
    "car_fuel_efficiency_2026.csv"
)

df = pd.read_csv(url)
print("Original dataset shape:", df.shape)

columns = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
    "fuel_efficiency_mpg",
]

df = df[columns].copy()
print("Selected dataset shape:", df.shape)
print(df.head())


# EDA: The distribution has no obvious long right tail.
plt.figure(figsize=(8, 4))
plt.hist(df["fuel_efficiency_mpg"], bins=50, edgecolor="white")
plt.xlabel("Fuel efficiency (mpg)")
plt.ylabel("Number of cars")
plt.title("Distribution of fuel efficiency")
plt.tight_layout()


# Question 1: Find the column with missing values.
print("\nQuestion 1 - Missing values:")
print(df.isna().sum())


# Question 2: Calculate the median horsepower.
print("\nQuestion 2 - Median horsepower:")
print(df["horsepower"].median())


# Split the data into 60% training, 20% validation, and 20% test.
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]].copy()
df_val = df.iloc[idx[n_train:n_train + n_val]].copy()
df_test = df.iloc[idx[n_train + n_val:]].copy()

print("\nTraining shape:", df_train.shape)
print("Validation shape:", df_val.shape)
print("Test shape:", df_test.shape)

# Separate the target from the input features.
y_train = df_train["fuel_efficiency_mpg"].to_numpy()
y_val = df_val["fuel_efficiency_mpg"].to_numpy()

df_train = df_train.drop(columns=["fuel_efficiency_mpg"])
df_val = df_val.drop(columns=["fuel_efficiency_mpg"])


# Question 3: Compare zero imputation with mean imputation.
# Calculate the mean using only the training data.
horsepower_mean = df_train["horsepower"].mean()

print("\nQuestion 3 - Missing value imputation:")
print("Training horsepower mean:", horsepower_mean)

fill_options = [
    ("Fill with 0", 0),
    ("Fill with mean", horsepower_mean),
]

for name, fill_value in fill_options:
    X_train = df_train.fillna(fill_value).to_numpy()
    X_val = df_val.fillna(fill_value).to_numpy()

    w0, w = train_linear_regression(X_train, y_train)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)

    print(name, "- RMSE:", round(score, 3))


# Question 4: Compare regularization values using zero imputation.
X_train = df_train.fillna(0).to_numpy()
X_val = df_val.fillna(0).to_numpy()

print("\nQuestion 4 - Regularization:")

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)

    print(f"r={r}: RMSE={round(score, 4):.4f}")


# Question 5: Check how the split seed affects validation RMSE.
print("\nQuestion 5 - Different split seeds:")
scores = []

for seed in range(10):
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    train = df.iloc[idx[:n_train]].copy()
    val = df.iloc[idx[n_train:n_train + n_val]].copy()
    test = df.iloc[idx[n_train + n_val:]].copy()

    y_train_seed = train["fuel_efficiency_mpg"].to_numpy()
    y_val_seed = val["fuel_efficiency_mpg"].to_numpy()

    X_train_seed = (
        train.drop(columns=["fuel_efficiency_mpg"]).fillna(0).to_numpy()
    )
    X_val_seed = (
        val.drop(columns=["fuel_efficiency_mpg"]).fillna(0).to_numpy()
    )

    w0, w = train_linear_regression(X_train_seed, y_train_seed)
    y_pred = w0 + X_val_seed.dot(w)
    score = rmse(y_val_seed, y_pred)

    # Keep the unrounded scores for the standard deviation calculation.
    scores.append(score)
    print(f"seed={seed}: RMSE={score:.6f}")

std = np.std(scores)
print("Standard deviation:", round(std, 3))


# Question 6: Split with seed 9 and combine training and validation data.
np.random.seed(9)
idx = np.arange(n)
np.random.shuffle(idx)

train = df.iloc[idx[:n_train]].copy()
val = df.iloc[idx[n_train:n_train + n_val]].copy()
test = df.iloc[idx[n_train + n_val:]].copy()

full_train = pd.concat([train, val])

y_full_train = full_train["fuel_efficiency_mpg"].to_numpy()
y_test_final = test["fuel_efficiency_mpg"].to_numpy()

X_full_train = (
    full_train.drop(columns=["fuel_efficiency_mpg"]).fillna(0).to_numpy()
)
X_test_final = (
    test.drop(columns=["fuel_efficiency_mpg"]).fillna(0).to_numpy()
)

# Train with zero imputation and r=0.001, then evaluate on the test set.
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
y_pred = w0 + X_test_final.dot(w)
test_score = rmse(y_test_final, y_pred)

print("\nQuestion 6 - Test RMSE:", round(test_score, 3))

# Show the histogram after printing all results.
plt.show()
