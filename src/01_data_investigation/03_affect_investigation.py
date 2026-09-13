from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "data/raw/2012-2013-data-with-predictions-4-final/"
    "2012-2013-data-with-predictions-4-final.csv"
)

df = pd.read_csv(DATA_PATH, nrows=100000)


affect_columns = [
    "Average_confidence(FRUSTRATED)",
    "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)",
    "Average_confidence(BORED)"
]


print("========== UNIQUE AFFECT VALUES ==========")

for column in affect_columns:
    print(
        column,
        "->",
        df[column].nunique(),
        "unique values"
    )


print("\n========== MOST COMMON AFFECT VALUES ==========")

for column in affect_columns:

    print("\n", column)

    print(
        df[column]
        .value_counts()
        .head(10)
    )


print("\n========== AFFECT COMBINATIONS ==========")

affect_patterns = (
    df[affect_columns]
    .drop_duplicates()
)

print(
    "Unique affect combinations:",
    len(affect_patterns)
)

print("\nFirst 20 combinations:")

print(
    affect_patterns
    .head(20)
    .to_string(index=False)
)


df["affect_sum"] = df[affect_columns].sum(axis=1)

print("\n========== AFFECT SUM ==========")

print(
    df["affect_sum"].describe()
)

print(
    "\nRows where affect sum ≈ 1:",
    (
        (df["affect_sum"] - 1).abs() < 0.001
    ).sum()
)

print(
    "Rows where affect sum > 1:",
    (df["affect_sum"] > 1).sum()
)



print("\n========== AFFECT CORRELATION ==========")

print(
    df[affect_columns].corr()
)