from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "data/raw/2012-2013-data-with-predictions-4-final/"
    "2012-2013-data-with-predictions-4-final.csv"
)

OUTPUT_PATH = Path(
    "data/processed/clean_interactions.csv"
)


columns = [
    "problem_log_id",
    "problemlogid",
    "user_id",
    "problem_id",
    "skill",
    "skill_id",
    "correct",
    "hint_count",
    "attempt_count",
    "ms_first_response",
    "start_time",
    "end_time",
    "Average_confidence(FRUSTRATED)",
    "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)",
    "Average_confidence(BORED)",
]


print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=columns
)

print("Raw rows:", len(df))


print("\nProcessing timestamps...")

df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)

df["end_time"] = pd.to_datetime(
    df["end_time"],
    errors="coerce"
)

print("Invalid start_time:", df["start_time"].isna().sum())
print("Invalid end_time:", df["end_time"].isna().sum())

print("\nCleaning response time...")

negative_response = df["ms_first_response"] < 0

print(
    "Negative response times:",
    negative_response.sum()
)


df.loc[
    negative_response,
    "ms_first_response"
] = pd.NA



print("\nChecking temporal consistency...")

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



df["valid_duration"] = valid_times & ~start_after_end



print("\nChecking duplicates...")

duplicate_problem_log = df["problem_log_id"].duplicated()

duplicate_problemlog = df["problemlogid"].duplicated()

print(
    "Duplicate problem_log_id:",
    duplicate_problem_log.sum()
)

print(
    "Duplicate problemlogid:",
    duplicate_problemlog.sum()
)


print("\nCreating quality flags...")

df["valid_start_time"] = df["start_time"].notna()

df["valid_end_time"] = df["end_time"].notna()

df["valid_response_time"] = (
    df["ms_first_response"].notna()
    & (df["ms_first_response"] >= 0)
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)



print("\nSaving cleaned dataset...")

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Saved to:", OUTPUT_PATH)
print("Cleaned rows:", len(df))



print("\n========== CLEANING SUMMARY ==========")

print("Rows removed:", 0)

print(
    "Invalid start_time:",
    (~df["valid_start_time"]).sum()
)

print(
    "Invalid end_time:",
    (~df["valid_end_time"]).sum()
)

print(
    "Invalid duration:",
    (~df["valid_duration"]).sum()
)

print(
    "Invalid response time:",
    (~df["valid_response_time"]).sum()
)

print(
    "Missing skill:",
    df["skill"].isna().sum()
)

print(
    "Missing skill_id:",
    df["skill_id"].isna().sum()
)

print("\nCleaning complete.")