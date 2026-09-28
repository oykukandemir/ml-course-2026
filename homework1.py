import numpy as np
import pandas as pd


# Question 1: Pandas version

print("Q1 - Pandas version:")
print(pd.__version__)


# Load the dataset

data_url = (
    "https://raw.githubusercontent.com/"
    "DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)

df = pd.read_csv(data_url)


# Question 2: Number of records

number_of_records = df.shape[0]

print("\nQ2 - Number of records:")
print(number_of_records)


# Question 3: Number of fuel types

number_of_fuel_types = df["fuel_type"].nunique()

print("\nQ3 - Fuel types:")
print(df["fuel_type"].unique())

print("Number of fuel types:")
print(number_of_fuel_types)


# Question 4: Columns with missing values

missing_values = df.isna().sum()
columns_with_missing_values = (missing_values > 0).sum()

print("\nQ4 - Missing values in each column:")
print(missing_values)

print("Number of columns with missing values:")
print(columns_with_missing_values)


# Question 5: Maximum fuel efficiency of Asian cars

asian_cars = df[df["origin"] == "Asia"]

maximum_efficiency = (
    asian_cars["fuel_efficiency_mpg"].max()
)

print("\nQ5 - Maximum fuel efficiency of Asian cars:")
print(maximum_efficiency)


# Question 6: Median horsepower before and after filling missing values

median_before = df["horsepower"].median()
most_frequent_horsepower = df["horsepower"].mode().iloc[0]

filled_horsepower = df["horsepower"].fillna(
    most_frequent_horsepower
)

median_after = filled_horsepower.median()

print("\nQ6 - Median horsepower before filling:")
print(median_before)

print("Most frequent horsepower:")
print(most_frequent_horsepower)

print("Median horsepower after filling:")
print(median_after)

print("The median decreased:")
print(median_after < median_before)


# Question 7: Sum of weights

X = (
    df[df["origin"] == "Asia"]
    [["vehicle_weight", "model_year"]]
    .head(7)
    .to_numpy(dtype=float)
)

XTX = X.T @ X
XTX_inverse = np.linalg.inv(XTX)

y = np.array(
    [1100, 1300, 800, 900, 1000, 1100, 1200],
    dtype=float
)

w = XTX_inverse @ X.T @ y
sum_of_weights = w.sum()

print("\nQ7 - X:")
print(X)

print("Weights:")
print(w)

print("Sum of weights:")
print(sum_of_weights)