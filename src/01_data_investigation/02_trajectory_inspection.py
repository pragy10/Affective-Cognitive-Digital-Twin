from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "data/raw/2012-2013-data-with-predictions-4-final/"
    "2012-2013-data-with-predictions-4-final.csv"
)


df = pd.read_csv(DATA_PATH, nrows=100000)

print("Loaded rows:", len(df))


df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)

df["end_time"] = pd.to_datetime(
    df["end_time"],
    errors="coerce"
)


print("\n========== TIME INFORMATION ==========")

print("Invalid start times:",
      df["start_time"].isna().sum())

print("Invalid end times:",
      df["end_time"].isna().sum())

print("Earliest interaction:",
      df["start_time"].min())

print("Latest interaction:",
      df["start_time"].max())

print(
    "\nRows globally chronological:",
    df["start_time"].is_monotonic_increasing
)


student_counts = df["user_id"].value_counts()

print("\n========== STUDENT INTERACTION COUNTS ==========")

print("Number of students:", len(student_counts))

print("Minimum interactions:",
      student_counts.min())

print("Maximum interactions:",
      student_counts.max())

print("Median interactions:",
      student_counts.median())

print("Mean interactions:",
      student_counts.mean())


student_id = student_counts.index[0]

student_df = df[
    df["user_id"] == student_id
].copy()

student_df = student_df.sort_values("start_time")


print("\n========== STUDENT TRAJECTORY ==========")

print("Selected student:", student_id)

print("Number of interactions:",
      len(student_df))


columns = [
    "user_id",
    "problem_id",
    "skill",
    "skill_id",
    "correct",
    "hint_count",
    "attempt_count",
    "ms_first_response",
    "start_time",
    "Average_confidence(FRUSTRATED)",
    "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)",
    "Average_confidence(BORED)"
]


print("\nFirst 20 interactions:")

print(
    student_df[columns]
    .head(20)
    .to_string(index=False)
)


affect_columns = [
    "Average_confidence(FRUSTRATED)",
    "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)",
    "Average_confidence(BORED)"
]


print("\n========== AFFECT STATISTICS ==========")

print(
    df[affect_columns].describe()
)


print("\n========== CORRECTNESS ==========")

print(
    df["correct"].value_counts(
        dropna=False
    )
)


behavior_columns = [
    "hint_count",
    "attempt_count",
    "ms_first_response",
    "bottom_hint"
]


print("\n========== BEHAVIORAL STATISTICS ==========")

print(
    df[behavior_columns].describe()
)