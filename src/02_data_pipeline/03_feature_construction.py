from pathlib import Path
import pandas as pd
import numpy as np






INPUT_PATH = Path(
    "data/processed/clean_interactions.csv"
)

OUTPUT_PATH = Path(
    "data/processed/featured_interactions.csv"
)






print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_PATH)

print("Rows:", len(df))






print("\nProcessing timestamps...")

df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)






print("\nSorting student trajectories...")

df = df.sort_values(
    ["user_id", "start_time"],
    kind="mergesort"
).reset_index(drop=True)

print("Sorting complete.")






print("\nCreating hint/attempt ratio...")

df["hint_attempt_ratio"] = np.where(
    df["attempt_count"] > 0,
    df["hint_count"] / df["attempt_count"],
    np.nan
)






print("Creating time gap...")

df["time_gap_seconds"] = (
    df.groupby("user_id")["start_time"]
    .diff()
    .dt.total_seconds()
)









print("Creating wrong-answer streak...")

is_wrong = (
    df["correct"] == 0
)


correct_group = (
    df.groupby("user_id")["correct"]
    .transform(
        lambda x: x.ne(0).cumsum()
    )
)

df["wrong_answer_streak"] = (
    is_wrong
    .groupby(
        [df["user_id"], correct_group]
    )
    .cumsum()
)






print("Creating response time in seconds...")

df["response_time_seconds"] = (
    df["ms_first_response"] / 1000.0
)






print("\n========== FEATURE SUMMARY ==========")

feature_columns = [
    "correct",
    "hint_count",
    "attempt_count",
    "hint_attempt_ratio",
    "response_time_seconds",
    "time_gap_seconds",
    "wrong_answer_streak",
]

for column in feature_columns:

    print(f"\n{column}")

    print(
        df[column].describe()
    )






print("\n========== FEATURE MISSING VALUES ==========")

for column in feature_columns:

    missing = df[column].isna().sum()

    print(
        f"{column}: "
        f"{missing:,}"
    )






print("\n========== WRONG STREAK CHECK ==========")

print(
    "Maximum wrong-answer streak:",
    df["wrong_answer_streak"].max()
)

print(
    "Rows with wrong-answer streak > 0:",
    (df["wrong_answer_streak"] > 0).sum()
)






print("\n========== TIME GAP CHECK ==========")

print(
    "Negative time gaps:",
    (df["time_gap_seconds"] < 0).sum()
)

print(
    "Zero time gaps:",
    (df["time_gap_seconds"] == 0).sum()
)






OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving featured dataset...")

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "Saved to:",
    OUTPUT_PATH
)

print(
    "Rows saved:",
    len(df)
)

print("\nFeature construction complete.")