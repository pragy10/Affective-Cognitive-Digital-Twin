from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/2012-2013-data-with-predictions-4-final")

csv_files = list(DATA_DIR.glob("*.csv"))

print("CSV files found:")
for file in csv_files:
    print(" -", file)

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found inside {DATA_DIR}"
    )

dataset_path = csv_files[0]

print("\nInspecting:", dataset_path)

df = pd.read_csv(
    dataset_path,
    nrows=10000
)

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
for column in df.columns:
    print(column)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== UNIQUE VALUES ==========")
for column in ["user_id", "problem_id", "skill_id"]:
    if column in df.columns:
        print(
            f"{column}:",
            df[column].nunique()
        )