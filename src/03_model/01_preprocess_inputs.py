from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEQUENCE_LENGTH = 50

MODEL_SEQUENCE_PATH = Path(
    "data/processed/model_sequences.npz"
)

FEATURED_DATA_PATH = Path(
    "data/processed/featured_interactions.csv"
)

OUTPUT_PATH = Path(
    "data/processed/model_inputs.npz"
)

STATS_PATH = Path(
    "data/processed/preprocessing_stats.npz"
)


BASE_FEATURES = [
    "correct",
    "hint_count",
    "attempt_count",
    "hint_attempt_ratio",
    "response_time_seconds",
    "time_gap_seconds",
    "wrong_answer_streak",
    "affect_frustrated",
    "affect_confused",
    "affect_concentrating",
    "affect_bored",
]


MISSING_FEATURES = [
    "hint_attempt_ratio",
    "response_time_seconds",
    "time_gap_seconds",
]


# ---------------------------------------------------------
# Load model sequences
# ---------------------------------------------------------

print("========== LOADING MODEL SEQUENCES ==========")

data = np.load(MODEL_SEQUENCE_PATH)

X = data["X"]
mask = data["mask"]
targets = data["targets"]
target_mask = data["target_mask"]

problem_ids = data["problem_ids"]
skill_ids = data["skill_ids"]
user_ids = data["user_ids"]

print("X shape:", X.shape)
print("Mask shape:", mask.shape)
print("Number of sequences:", len(user_ids))


# ---------------------------------------------------------
# Load featured interaction data
# ---------------------------------------------------------

print("\n========== LOADING FEATURED DATA ==========")

df = pd.read_csv(FEATURED_DATA_PATH)

print("Rows:", len(df))

required_columns = [
    "user_id",
    "start_time",
] + [
    feature
    for feature in BASE_FEATURES
    if feature not in [
        "affect_frustrated",
        "affect_confused",
        "affect_concentrating",
        "affect_bored",
    ]
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

print("Required columns: OK")


# ---------------------------------------------------------
# Reproduce original ordering
# ---------------------------------------------------------

print("\n========== REPRODUCING TRAJECTORY ORDER ==========")

df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)

df = df.sort_values(
    ["user_id", "start_time"],
    kind="mergesort"
).reset_index(drop=True)

print("Sorting complete.")


# ---------------------------------------------------------
# Verify sequence count
# ---------------------------------------------------------

expected_sequences = 0

for _, student_df in df.groupby(
    "user_id",
    sort=False
):

    length = len(student_df)

    expected_sequences += int(
        np.ceil(length / SEQUENCE_LENGTH)
    )


print(
    "Expected sequences from original data:",
    expected_sequences
)

print(
    "Sequences in model file:",
    len(X)
)

if expected_sequences != len(X):

    raise ValueError(
        "Sequence count does not match the original "
        "sequence construction."
    )

print("Sequence count: OK")


# ---------------------------------------------------------
# Build missingness indicators
# ---------------------------------------------------------

print("\n========== BUILDING MISSINGNESS INDICATORS ==========")

missing_sequences = []

sequence_user_ids = []


for user_id, student_df in df.groupby(
    "user_id",
    sort=False
):

    missing_values = (
        student_df[MISSING_FEATURES]
        .isna()
        .to_numpy(dtype=np.float32)
    )

    length = len(missing_values)

    for start in range(
        0,
        length,
        SEQUENCE_LENGTH
    ):

        window = missing_values[
            start:start + SEQUENCE_LENGTH
        ]

        actual_length = len(window)

        if actual_length < SEQUENCE_LENGTH:

            padding_rows = (
                SEQUENCE_LENGTH
                - actual_length
            )

            padding = np.zeros(
                (
                    padding_rows,
                    len(MISSING_FEATURES)
                ),
                dtype=np.float32
            )

            window = np.vstack(
                [window, padding]
            )

        missing_sequences.append(window)

        sequence_user_ids.append(user_id)


missing_indicators = np.asarray(
    missing_sequences,
    dtype=np.float32
)

sequence_user_ids = np.asarray(
    sequence_user_ids
)


print(
    "Missing indicator shape:",
    missing_indicators.shape
)


# ---------------------------------------------------------
# Verify ordering against model sequences
# ---------------------------------------------------------

print("\n========== VERIFYING SEQUENCE ORDER ==========")

if not np.array_equal(
    sequence_user_ids,
    user_ids
):

    raise ValueError(
        "Sequence user ordering does not match "
        "model_sequences.npz."
    )

print("Sequence user ordering: OK")


# ---------------------------------------------------------
# Verify missingness indicators
# ---------------------------------------------------------

print("\n========== MISSINGNESS STATISTICS ==========")

for i, feature in enumerate(MISSING_FEATURES):

    count = int(
        missing_indicators[:, :, i].sum()
    )

    print(
        f"{feature} missing values:",
        count
    )


# ---------------------------------------------------------
# Copy X before transformation
# ---------------------------------------------------------

X_processed = X.copy()


# ---------------------------------------------------------
# Apply log1p transformations
# ---------------------------------------------------------

print("\n========== APPLYING LOG TRANSFORMATIONS ==========")

LOG_FEATURES = [
    "hint_count",
    "attempt_count",
    "hint_attempt_ratio",
    "response_time_seconds",
    "time_gap_seconds",
    "wrong_answer_streak",
]


feature_index = {
    feature: i
    for i, feature in enumerate(BASE_FEATURES)
}


for feature in LOG_FEATURES:

    idx = feature_index[feature]

    values = X_processed[:, :, idx]

    # All these features are non-negative after
    # Phase 2 cleaning.

    if np.any(values < 0):

        raise ValueError(
            f"Negative values found in {feature}"
        )

    X_processed[:, :, idx] = np.log1p(values)

    print(
        f"{feature}: log1p applied"
    )


# ---------------------------------------------------------
# Identify training sequences
# ---------------------------------------------------------

print("\n========== IDENTIFYING TRAINING SEQUENCES ==========")

split_data = np.load(
    "data/processed/student_split.npz"
)

train_user_ids = split_data["train_users"]

train_user_set = set(
    train_user_ids.tolist()
)

train_sequence_mask = np.array(
    [
        user_id in train_user_set
        for user_id in user_ids
    ],
    dtype=bool
)

print(
    "Training sequences:",
    train_sequence_mask.sum()
)

print(
    "Non-training sequences:",
    (~train_sequence_mask).sum()
)


# ---------------------------------------------------------
# Calculate training statistics
# ---------------------------------------------------------

print("\n========== CALCULATING TRAINING STATISTICS ==========")

# Features that will be standardized.
#
# correct and affect values stay in their natural
# 0-1 scale.

STANDARDIZE_FEATURES = [
    "hint_count",
    "attempt_count",
    "hint_attempt_ratio",
    "response_time_seconds",
    "time_gap_seconds",
    "wrong_answer_streak",
]


means = {}
stds = {}


for feature in STANDARDIZE_FEATURES:

    idx = feature_index[feature]

    values = X_processed[
        train_sequence_mask,
        :,
        idx
    ]

    valid_values = values[
        mask[train_sequence_mask] == 1
    ]

    mean = float(
        np.mean(valid_values)
    )

    std = float(
        np.std(valid_values)
    )

    if std == 0:

        raise ValueError(
            f"Zero standard deviation for {feature}"
        )

    means[feature] = mean
    stds[feature] = std

    print(
        f"{feature}: "
        f"mean={mean:.6f}, "
        f"std={std:.6f}"
    )


# ---------------------------------------------------------
# Standardize using training statistics
# ---------------------------------------------------------

print("\n========== STANDARDIZING FEATURES ==========")

for feature in STANDARDIZE_FEATURES:

    idx = feature_index[feature]

    X_processed[:, :, idx] = (
        X_processed[:, :, idx]
        - means[feature]
    ) / stds[feature]

    print(
        f"{feature}: standardized"
    )


# ---------------------------------------------------------
# Add missingness indicators
# ---------------------------------------------------------

print("\n========== ADDING MISSINGNESS FEATURES ==========")

X_processed = np.concatenate(
    [
        X_processed,
        missing_indicators
    ],
    axis=2
)

X_processed[mask == 0] = 0.0

print(
    "Original feature count:",
    len(BASE_FEATURES)
)

print(
    "Missingness feature count:",
    len(MISSING_FEATURES)
)

print(
    "Final feature count:",
    X_processed.shape[2]
)


# ---------------------------------------------------------
# Validate
# ---------------------------------------------------------

print("\n========== FINAL VALIDATION ==========")

print(
    "X processed shape:",
    X_processed.shape
)

print(
    "NaN values:",
    np.isnan(X_processed).sum()
)

print(
    "Infinite values:",
    np.isinf(X_processed).sum()
)

print(
    "Expected feature count:",
    len(BASE_FEATURES)
    + len(MISSING_FEATURES)
)

if np.isnan(X_processed).any():

    raise ValueError(
        "NaN values found after preprocessing."
    )

if np.isinf(X_processed).any():

    raise ValueError(
        "Infinite values found after preprocessing."
    )

if X_processed.shape[2] != 14:

    raise ValueError(
        "Expected 14 input features."
    )


# Padding must remain zero for the original
# processed features and missingness indicators.

padding_values = X_processed[
    mask == 0
]

print(
    "Maximum absolute padding value:",
    np.max(np.abs(padding_values))
)


# ---------------------------------------------------------
# Save processed inputs
# ---------------------------------------------------------

print("\n========== SAVING MODEL INPUTS ==========")

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

np.savez_compressed(
    OUTPUT_PATH,
    X=X_processed.astype(np.float32),
    mask=mask,
    targets=targets,
    target_mask=target_mask,
    problem_ids=problem_ids,
    skill_ids=skill_ids,
    user_ids=user_ids,
)

print(
    "Saved:",
    OUTPUT_PATH
)


# ---------------------------------------------------------
# Save preprocessing statistics
# ---------------------------------------------------------

np.savez(
    STATS_PATH,
    feature_names=np.array(
        BASE_FEATURES
        + MISSING_FEATURES
    ),
    standardized_features=np.array(
        STANDARDIZE_FEATURES
    ),
    means=np.array(
        [
            means[f]
            for f in STANDARDIZE_FEATURES
        ],
        dtype=np.float32
    ),
    stds=np.array(
        [
            stds[f]
            for f in STANDARDIZE_FEATURES
        ],
        dtype=np.float32
    ),
)

print(
    "Saved preprocessing statistics:",
    STATS_PATH
)


print("\n========== PHASE 3.1 COMPLETE ==========")