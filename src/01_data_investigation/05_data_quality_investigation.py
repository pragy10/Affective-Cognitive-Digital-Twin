from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "data/raw/2012-2013-data-with-predictions-4-final/"
    "2012-2013-data-with-predictions-4-final.csv"
)


columns = [
    "problem_log_id",
    "problemlogid",
    "user_id",
    "skill",
    "skill_id",
    "correct",
    "hint_count",
    "attempt_count",
    "ms_first_response",
    "start_time",
    "end_time",
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

df["end_time"] = pd.to_datetime(
    df["end_time"],
    errors="coerce"
)



print("\n========== MISSING VALUES ==========")

missing = df.isna().sum()

print(
    missing[missing > 0]
)



print("\n========== TIMESTAMP QUALITY ==========")

invalid_start = df["start_time"].isna()
invalid_end = df["end_time"].isna()

print(
    "Invalid start_time:",
    invalid_start.sum()
)

print(
    "Invalid end_time:",
    invalid_end.sum()
)


valid_times = (
    df["start_time"].notna()
    & df["end_time"].notna()
)

start_after_end = (
    valid_times
    & (df["start_time"] > df["end_time"])
)

print(
    "start_time > end_time:",
    start_after_end.sum()
)



print("\n========== DUPLICATE IDENTIFIERS ==========")

print(
    "Duplicate problem_log_id rows:",
    df["problem_log_id"].duplicated().sum()
)

print(
    "Duplicate problemlogid rows:",
    df["problemlogid"].duplicated().sum()
)


print("\n========== CORRECTNESS ==========")

print(
    "Unique correctness values:"
)

print(
    sorted(
        df["correct"]
        .dropna()
        .unique()
        .tolist()
    )
)

print("\nCorrectness value counts:")

print(
    df["correct"]
    .value_counts(dropna=False)
)


print("\n========== ATTEMPT COUNT ==========")

print(
    df["attempt_count"].describe()
)

print(
    "Zero attempts:",
    (df["attempt_count"] == 0).sum()
)

print(
    "Negative attempts:",
    (df["attempt_count"] < 0).sum()
)


print("\n========== HINT COUNT ==========")

print(
    df["hint_count"].describe()
)

print(
    "Negative hints:",
    (df["hint_count"] < 0).sum()
)


print("\n========== FIRST RESPONSE TIME ==========")

print(
    df["ms_first_response"].describe()
)

print(
    "Zero response time:",
    (df["ms_first_response"] == 0).sum()
)

print(
    "Negative response time:",
    (df["ms_first_response"] < 0).sum()
)

print(
    "Response time > 5 minutes:",
    (df["ms_first_response"] > 5 * 60 * 1000).sum()
)

print(
    "Response time > 30 minutes:",
    (df["ms_first_response"] > 30 * 60 * 1000).sum()
)

print(
    "Response time > 1 hour:",
    (df["ms_first_response"] > 60 * 60 * 1000).sum()
)


print("\n========== SKILL INFORMATION ==========")

print(
    "Missing skill:",
    df["skill"].isna().sum()
)

print(
    "Missing skill_id:",
    df["skill_id"].isna().sum()
)

print(
    "Both skill and skill_id missing:",
    (
        df["skill"].isna()
        & df["skill_id"].isna()
    ).sum()
)

print(
    "Skill present but skill_id missing:",
    (
        df["skill"].notna()
        & df["skill_id"].isna()
    ).sum()
)

print(
    "Skill missing but skill_id present:",
    (
        df["skill"].isna()
        & df["skill_id"].notna()
    ).sum()
)


print("\n========== INVESTIGATION SUMMARY ==========")

print("Total rows:", len(df))

print(
    "Rows with invalid timestamps:",
    (
        invalid_start
        | invalid_end
    ).sum()
)

print(
    "Rows with start_time > end_time:",
    start_after_end.sum()
)

print(
    "Rows with duplicate problem_log_id:",
    df["problem_log_id"].duplicated().sum()
)

print(
    "Rows with negative attempts:",
    (df["attempt_count"] < 0).sum()
)

print(
    "Rows with negative hints:",
    (df["hint_count"] < 0).sum()
)

print(
    "Rows with negative response time:",
    (df["ms_first_response"] < 0).sum()
)