from pathlib import Path
import pandas as pd
import numpy as np


DATA_PATH = Path(
    "data/raw/2012-2013-data-with-predictions-4-final/"
    "2012-2013-data-with-predictions-4-final.csv"
)

columns = [
    "user_id",
    "problem_id",
    "skill_id",
    "correct",
    "hint_count",
    "attempt_count",
    "ms_first_response",
    "start_time",
]


print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=columns
)

print("Total rows:", len(df))






df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)

df = df.sort_values(
    ["user_id", "start_time"]
)

print("\n========== TIME GAP ==========")

df["time_gap_seconds"] = (
    df.groupby("user_id")["start_time"]
    .diff()
    .dt.total_seconds()
)

valid_gap = df["time_gap_seconds"].dropna()

print(valid_gap.describe())

print("\nTime gaps:")
print("Negative gaps:", (valid_gap < 0).sum())
print("Zero gaps:", (valid_gap == 0).sum())
print("> 1 minute:", (valid_gap > 60).sum())
print("> 5 minutes:", (valid_gap > 5 * 60).sum())
print("> 10 minutes:", (valid_gap > 10 * 60).sum())
print("> 30 minutes:", (valid_gap > 30 * 60).sum())
print("> 1 hour:", (valid_gap > 60 * 60).sum())






print("\n========== WRONG-ANSWER STREAKS ==========")

df["is_wrong"] = (df["correct"] == 0).astype(int)


df["correct_group"] = (
    df.groupby("user_id")["is_wrong"]
    .transform(lambda x: x.eq(0).cumsum())
)

wrong_streak = (
    df[df["is_wrong"] == 1]
    .groupby(["user_id", "correct_group"])
    .size()
)

print("Maximum wrong-answer streak:", wrong_streak.max())
print("Mean wrong-answer streak:", wrong_streak.mean())
print("Median wrong-answer streak:", wrong_streak.median())

print("\nStreak distribution:")
print(
    wrong_streak.value_counts()
    .sort_index()
    .head(15)
)

print("\nStudents with streak >= 3:",
      (wrong_streak >= 3).sum())

print("Students with streak >= 5:",
      (wrong_streak >= 5).sum())






print("\n========== HINT / ATTEMPT RATIO ==========")

df["hint_attempt_ratio"] = np.where(
    df["attempt_count"] > 0,
    df["hint_count"] / df["attempt_count"],
    np.nan
)

ratio = df["hint_attempt_ratio"].dropna()

print(ratio.describe())

print("\nRatio > 1:", (ratio > 1).sum())
print("Ratio > 2:", (ratio > 2).sum())
print("Ratio > 5:", (ratio > 5).sum())






print("\n========== RESPONSE TIME ==========")

response = df["ms_first_response"]

valid_response = response[
    response >= 0
].dropna()

print("Valid response times:", len(valid_response))

print("\nPercentiles:")
print(
    valid_response.quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    )
)

print("\nResponse time thresholds:")

for seconds in [10, 30, 60, 120, 300, 600]:
    count = (valid_response > seconds * 1000).sum()
    percentage = count / len(valid_response) * 100

    print(
        f">{seconds} sec: "
        f"{count:,} ({percentage:.2f}%)"
    )






print("\n========== WITHIN-STUDENT RESPONSE-TIME VARIATION ==========")

df["response_seconds"] = df["ms_first_response"] / 1000

valid_response_df = df[
    df["response_seconds"] >= 0
].copy()

student_response_stats = (
    valid_response_df
    .groupby("user_id")["response_seconds"]
    .agg(
        count="count",
        mean="mean",
        std="std"
    )
)

students_with_variation = student_response_stats[
    student_response_stats["count"] >= 3
]

print(
    "Students with >=3 valid response times:",
    len(students_with_variation)
)

print("\nWithin-student standard deviation:")
print(
    students_with_variation["std"].describe()
)






print("\n========== CORRECTNESS VARIATION ==========")

print(
    df["correct"]
    .value_counts()
    .sort_index()
)






print("\n========== FEATURE CORRELATIONS ==========")

feature_columns = [
    "correct",
    "hint_count",
    "attempt_count",
    "ms_first_response",
]

print(
    df[feature_columns]
    .corr()
    .round(3)
)






print("\n========== FEATURE AVAILABILITY ==========")

for column in [
    "correct",
    "hint_count",
    "attempt_count",
    "ms_first_response",
    "start_time",
    "user_id",
]:

    available = df[column].notna().sum()
    percentage = available / len(df) * 100

    print(
        f"{column}: "
        f"{available:,} / {len(df):,} "
        f"({percentage:.2f}%)"
    )






print("\n========== INVESTIGATION SUMMARY ==========")

print("Rows:", len(df))

print(
    "Valid time gaps:",
    len(valid_gap)
)

print(
    "Wrong-answer streaks:",
    len(wrong_streak)
)

print(
    "Valid hint/attempt ratios:",
    len(ratio)
)

print(
    "Valid response times:",
    len(valid_response)
)

print(
    "Students with >=3 response times:",
    len(students_with_variation)
)

print("\nCandidate behavioral features:")
print("1. Correctness")
print("2. Hint count")
print("3. Attempt count")
print("4. Hint/attempt ratio")
print("5. Response time")
print("6. Inter-action time gap")
print("7. Wrong-answer streak")
print("8. Within-student response-time variation")