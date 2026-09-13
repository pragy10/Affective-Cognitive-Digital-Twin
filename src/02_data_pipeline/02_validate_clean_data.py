from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "data/processed/clean_interactions.csv"
)


print("Loading cleaned dataset...")

df = pd.read_csv(DATA_PATH)

print("Rows:", len(df))
print("Columns:", len(df.columns))


print("\n========== REQUIRED COLUMNS ==========")

required_columns = [
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
    "valid_duration",
    "valid_start_time",
    "valid_end_time",
    "valid_response_time",
]

for column in required_columns:
    print(
        f"{column}:",
        "OK" if column in df.columns else "MISSING"
    )


print("\n========== ROW COUNT ==========")

print("Expected:", 6_123_270)
print("Actual:  ", len(df))
print(
    "PASS"
    if len(df) == 6_123_270
    else "FAIL"
)


print("\n========== RESPONSE TIME ==========")

print(
    "Negative response times:",
    (df["ms_first_response"] < 0).sum()
)

print(
    "Missing response times:",
    df["ms_first_response"].isna().sum()
)


print("\n========== TIMESTAMP FLAGS ==========")

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


print("\n========== DUPLICATES ==========")

print(
    "Duplicate problem_log_id:",
    df["problem_log_id"].duplicated().sum()
)

print(
    "Duplicate problemlogid:",
    df["problemlogid"].duplicated().sum()
)


print("\n========== CORRECTNESS ==========")

print(
    sorted(
        df["correct"]
        .dropna()
        .unique()
        .tolist()
    )
)


print("\n========== AFFECT ==========")

affect_columns = [
    "Average_confidence(FRUSTRATED)",
    "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)",
    "Average_confidence(BORED)",
]

for column in affect_columns:
    print(
        column,
        "missing:",
        df[column].isna().sum()
    )


print("\n========== SKILL ==========")

print(
    "Missing skill:",
    df["skill"].isna().sum()
)

print(
    "Missing skill_id:",
    df["skill_id"].isna().sum()
)


print("\n========== VALIDATION COMPLETE ==========")