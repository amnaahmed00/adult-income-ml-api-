
from src.adult_income.data import load_raw_data

train_df, test_df = load_raw_data()

print("\n--- Training dataset shape ---")
print(train_df.shape)

print("\n--- Test dataset shape ---")
print(test_df.shape)

print("\n--- First five training records ---")
print(train_df.head())

print("\n--- Missing values in training data ---")
print(train_df.isna().sum().sort_values(ascending=False))

print("\n--- Target distribution in training data ---")
print(train_df["income"].value_counts())

print("\n--- Target proportions in training data ---")
print(train_df["income"].value_counts(normalize=True).round(3))

print("\n--- Data types ---")
print(train_df.dtypes)
