from pathlib import Path
import pandas as pd
import numpy as np






SEQUENCE_LENGTH = 50

INPUT_PATH = Path(
    "data/processed/affect_interactions.csv"
)

OUTPUT_PATH = Path(
    "data/processed/sequences.npz"
)






BEHAVIOR_FEATURES = [
    "correct",
    "hint_count",
    "attempt_count",
    "hint_attempt_ratio",
    "response_time_seconds",
    "time_gap_seconds",
    "wrong_answer_streak",
]

AFFECT_FEATURES = [
    "affect_frustrated",
    "affect_confused",
    "affect_concentrating",
    "affect_bored",
]

SEQUENCE_FEATURES = (
    BEHAVIOR_FEATURES +
    AFFECT_FEATURES
)






print("Loading affect-enhanced dataset...")

df = pd.read_csv(INPUT_PATH)

print("Rows:", len(df))






print("\n========== COLUMN VALIDATION ==========")

required_columns = [
    "user_id",
    "problem_id",
    "skill",
    "skill_id",
] + SEQUENCE_FEATURES

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

    print(f"{column}: OK")






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






print("\n========== TRAJECTORY STATISTICS ==========")

student_lengths = (
    df.groupby("user_id")
    .size()
)

print(
    "Number of students:",
    len(student_lengths)
)

print(
    "Minimum trajectory length:",
    student_lengths.min()
)

print(
    "Median trajectory length:",
    student_lengths.median()
)

print(
    "Mean trajectory length:",
    student_lengths.mean()
)

print(
    "Maximum trajectory length:",
    student_lengths.max()
)






short_students = student_lengths[
    student_lengths < SEQUENCE_LENGTH
]

full_or_long_students = student_lengths[
    student_lengths >= SEQUENCE_LENGTH
]

print("\n========== WINDOW STATISTICS ==========")

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Students shorter than sequence length:",
    len(short_students)
)

print(
    "Students with at least one full window:",
    len(full_or_long_students)
)

print(
    "Interactions belonging to short students:",
    short_students.sum()
)






print("\nCreating padded sequence windows...")

sequences = []
masks = []
sequence_user_ids = []




PADDING_VALUE = 0.0


for user_id, student_df in df.groupby(
    "user_id",
    sort=False
):

    values = student_df[
        SEQUENCE_FEATURES
    ].to_numpy(dtype=np.float32)

    length = len(values)

    
    
    

    for start in range(
        0,
        length,
        SEQUENCE_LENGTH
    ):

        window = values[
            start:start + SEQUENCE_LENGTH
        ]

        actual_length = len(window)

        
        
        
        

        window = np.nan_to_num(
            window,
            nan=PADDING_VALUE
        )

        
        
        

        if actual_length < SEQUENCE_LENGTH:

            padding_rows = (
                SEQUENCE_LENGTH -
                actual_length
            )

            padding = np.full(
                (
                    padding_rows,
                    len(SEQUENCE_FEATURES)
                ),
                PADDING_VALUE,
                dtype=np.float32
            )

            window = np.vstack(
                [window, padding]
            )

        
        
        
        
        
        

        mask = np.zeros(
            SEQUENCE_LENGTH,
            dtype=np.float32
        )

        mask[:actual_length] = 1.0

        
        
        

        sequences.append(window)
        masks.append(mask)
        sequence_user_ids.append(user_id)






print("\nConverting sequences to NumPy...")

X = np.asarray(
    sequences,
    dtype=np.float32
)

mask = np.asarray(
    masks,
    dtype=np.float32
)

sequence_user_ids = np.asarray(
    sequence_user_ids
)






print("\n========== SEQUENCE VALIDATION ==========")

print(
    "Number of sequences:",
    len(X)
)

print(
    "X shape:",
    X.shape
)

print(
    "Mask shape:",
    mask.shape
)

print(
    "Expected sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Expected feature count:",
    len(SEQUENCE_FEATURES)
)






if X.shape[1] != SEQUENCE_LENGTH:
    raise ValueError(
        "Incorrect sequence length."
    )

if X.shape[2] != len(SEQUENCE_FEATURES):
    raise ValueError(
        "Incorrect feature count."
    )

if mask.shape != (
    X.shape[0],
    SEQUENCE_LENGTH
):
    raise ValueError(
        "Incorrect mask shape."
    )






print("\n========== VALUE VALIDATION ==========")

print(
    "NaN values in X:",
    np.isnan(X).sum()
)

print(
    "Infinite values in X:",
    np.isinf(X).sum()
)

print(
    "NaN values in mask:",
    np.isnan(mask).sum()
)

print(
    "Infinite values in mask:",
    np.isinf(mask).sum()
)






print("\n========== MASK STATISTICS ==========")

real_timesteps = int(
    mask.sum()
)

total_timesteps = (
    mask.shape[0] *
    mask.shape[1]
)

padding_timesteps = (
    total_timesteps -
    real_timesteps
)

print(
    "Real timesteps:",
    real_timesteps
)

print(
    "Padding timesteps:",
    padding_timesteps
)

print(
    "Total timesteps:",
    total_timesteps
)

print(
    "Padding percentage:",
    f"{padding_timesteps / total_timesteps * 100:.2f}%"
)






print("\n========== USER COVERAGE ==========")

unique_users = np.unique(
    sequence_user_ids
)

print(
    "Unique users represented:",
    len(unique_users)
)

print(
    "Total users in original dataset:",
    len(student_lengths)
)

print(
    "Users missing from sequences:",
    len(student_lengths) -
    len(unique_users)
)






print("\n========== EXAMPLE MASKS ==========")

print(
    "First 5 sequence masks:"
)

for i in range(
    min(5, len(mask))
):

    print(
        f"Sequence {i}:",
        mask[i].astype(int)
    )






OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving sequences...")

np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    mask=mask,
    user_ids=sequence_user_ids,
)

print(
    "Saved to:",
    OUTPUT_PATH
)

print("\nSequence construction complete.")