from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "data/raw/2012-2013-data-with-predictions-4-final/"
    "2012-2013-data-with-predictions-4-final.csv"
)


print("Loading full dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=["user_id"]
)

print("Total rows:", len(df))


student_counts = df["user_id"].value_counts()

print("\n========== STUDENT SEQUENCE LENGTH ==========")

print("Unique students:", len(student_counts))

print("Minimum:", student_counts.min())
print("Maximum:", student_counts.max())
print("Mean:", student_counts.mean())
print("Median:", student_counts.median())

print("\nPercentiles:")

print(
    student_counts.quantile(
        [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)



print("\n========== SEQUENCE LENGTH THRESHOLDS ==========")

thresholds = [2, 5, 10, 20, 50, 100]

for threshold in thresholds:

    count = (student_counts >= threshold).sum()

    percentage = (
        count / len(student_counts)
    ) * 100

    print(
        f">= {threshold:3} interactions: "
        f"{count:6} students "
        f"({percentage:.2f}%)"
    )


print("\n========== SHORT SEQUENCES ==========")

for length in range(1, 11):

    count = (student_counts == length).sum()

    print(
        f"{length:2} interaction(s): "
        f"{count:6} students"
    )