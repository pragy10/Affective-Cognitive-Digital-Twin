import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/processed/affect_interactions.csv"
OUTPUT_FILE = "data/processed/model_sequences.npz"

SEQUENCE_LENGTH = 50

FEATURE_COLUMNS = [
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

REQUIRED_COLUMNS = [
    "user_id",
    "problem_id",
    "skill_id",
    "start_time",
    "correct",
] + [
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


# ============================================================
# Load data
# ============================================================

print("Loading affect-enriched interactions...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# Validate required columns
# ============================================================

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required columns found.")


# ============================================================
# Parse timestamp
# ============================================================

df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)


# ============================================================
# Sort interactions
# ============================================================

print("\nSorting interactions by student and time...")

df = df.sort_values(
    ["user_id", "start_time"],
    kind="mergesort"
).reset_index(drop=True)

print("Sorting complete.")


# ============================================================
# Containers
# ============================================================

X_sequences = []
mask_sequences = []

target_sequences = []
affect_target_sequences = []
target_mask_sequences = []

problem_id_sequences = []
skill_id_sequences = []
user_id_sequences = []


# ============================================================
# Feature / affect columns
# ============================================================

affect_columns = [
    "affect_frustrated",
    "affect_confused",
    "affect_concentrating",
    "affect_bored",
]


# ============================================================
# Build sequences
# ============================================================

print("\nBuilding sequences...")

sequence_count = 0
real_timesteps = 0


for user_id, group in df.groupby(
    "user_id",
    sort=False
):

    group = group.reset_index(drop=True)

    # --------------------------------------------------------
    # Process this student's interactions in chunks of 50
    # --------------------------------------------------------

    for start in range(
        0,
        len(group),
        SEQUENCE_LENGTH
    ):

        window = group.iloc[
            start:start + SEQUENCE_LENGTH
        ]

        actual_length = len(window)

        # ----------------------------------------------------
        # Feature values
        # ----------------------------------------------------

        feature_values = window[
            FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )

        feature_values = np.nan_to_num(
            feature_values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Pad feature matrix
        # ----------------------------------------------------

        X = np.zeros(
            (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)),
            dtype=np.float32
        )

        X[:actual_length] = feature_values

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        mask = np.zeros(
            SEQUENCE_LENGTH,
            dtype=np.float32
        )

        mask[:actual_length] = 1.0

        # ----------------------------------------------------
        # Problem IDs
        # ----------------------------------------------------

        problem_ids = window[
            "problem_id"
        ].fillna(
            ""
        ).astype(str).to_numpy()

        padded_problem_ids = np.full(
            SEQUENCE_LENGTH,
            "",
            dtype="<U64"
        )

        padded_problem_ids[:actual_length] = (
            problem_ids
        )

        # ----------------------------------------------------
        # Skill IDs
        # ----------------------------------------------------

        skill_ids = (
            pd.to_numeric(
                window["skill_id"],
                errors="coerce"
            )
            .fillna(-1)
            .to_numpy(dtype=np.int32)
        )

        padded_skill_ids = np.full(
            SEQUENCE_LENGTH,
            -1,
            dtype=np.int32
        )

        padded_skill_ids[:actual_length] = (
            skill_ids
        )

        # ----------------------------------------------------
        # Correctness targets
        #
        # At timestep t:
        # target[t] = correct[t+1]
        #
        # Last real timestep has no next interaction.
        # Padding also has no target.
        # ----------------------------------------------------

        correct_values = window[
            "correct"
        ].to_numpy(
            dtype=np.float32
        )

        correct_values = np.nan_to_num(
            correct_values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        targets = np.zeros(
            SEQUENCE_LENGTH,
            dtype=np.float32
        )

        if actual_length > 1:

            targets[:actual_length - 1] = (
                correct_values[1:]
            )

        # ----------------------------------------------------
        # Affect targets
        #
        # At timestep t:
        #
        # affect_targets[t] =
        #     affect[t+1]
        #
        # The 4 dimensions are:
        # [frustrated, confused,
        #  concentrating, bored]
        #
        # Last real timestep has no target.
        # Padding also has no target.
        # ----------------------------------------------------

        affect_values = window[
            affect_columns
        ].to_numpy(
            dtype=np.float32
        )

        affect_values = np.nan_to_num(
            affect_values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        affect_targets = np.zeros(
            (SEQUENCE_LENGTH, 4),
            dtype=np.float32
        )

        if actual_length > 1:

            affect_targets[:actual_length - 1] = (
                affect_values[1:]
            )

        # ----------------------------------------------------
        # Target mask
        #
        # 1 = there is a real next interaction
        # 0 = last interaction or padding
        # ----------------------------------------------------

        target_mask = np.zeros(
            SEQUENCE_LENGTH,
            dtype=np.float32
        )

        if actual_length > 1:

            target_mask[:actual_length - 1] = 1.0

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        X_sequences.append(X)
        mask_sequences.append(mask)

        target_sequences.append(targets)
        affect_target_sequences.append(
            affect_targets
        )
        target_mask_sequences.append(target_mask)

        problem_id_sequences.append(
            padded_problem_ids
        )

        skill_id_sequences.append(
            padded_skill_ids
        )

        user_id_sequences.append(
            user_id
        )

        sequence_count += 1
        real_timesteps += actual_length


# ============================================================
# Stack sequences
# ============================================================

print("\nStacking sequences...")

X = np.stack(
    X_sequences
)

mask = np.stack(
    mask_sequences
)

targets = np.stack(
    target_sequences
)

affect_targets = np.stack(
    affect_target_sequences
)

target_mask = np.stack(
    target_mask_sequences
)

problem_ids = np.stack(
    problem_id_sequences
)

skill_ids = np.stack(
    skill_id_sequences
)

user_ids = np.array(
    user_id_sequences
)


# ============================================================
# Validation
# ============================================================

print("\n--- Final Shapes ---")

print(
    f"X shape:             {X.shape}"
)

print(
    f"Mask shape:          {mask.shape}"
)

print(
    f"Targets shape:       {targets.shape}"
)

print(
    f"Affect targets shape:{affect_targets.shape}"
)

print(
    f"Target mask shape:   {target_mask.shape}"
)

print(
    f"Problem IDs shape:   {problem_ids.shape}"
)

print(
    f"Skill IDs shape:     {skill_ids.shape}"
)

print(
    f"User IDs shape:      {user_ids.shape}"
)


# ============================================================
# Basic shape checks
# ============================================================

assert X.shape[:2] == mask.shape

assert X.shape[:2] == targets.shape

assert X.shape[:2] == affect_targets.shape[:2]

assert affect_targets.shape[2] == 4

assert X.shape[:2] == target_mask.shape

assert X.shape[:2] == problem_ids.shape

assert X.shape[:2] == skill_ids.shape

assert X.shape[0] == user_ids.shape[0]


# ============================================================
# Target mask checks
# ============================================================

assert np.all(
    target_mask <= mask
)


# No targets on padding
assert np.all(
    target_mask[mask == 0] == 0
)


# ============================================================
# Padding checks
# ============================================================

assert np.all(
    X[mask == 0] == 0
)

assert np.all(
    targets[target_mask == 0] == 0
)

assert np.all(
    affect_targets[target_mask == 0] == 0
)


# ============================================================
# Affect target range
# ============================================================

real_affect_targets = affect_targets[
    target_mask == 1
]

assert np.all(
    real_affect_targets >= 0
)

assert np.all(
    real_affect_targets <= 1
)


# ============================================================
# Finite value checks
# ============================================================

assert np.isfinite(
    X
).all()

assert np.isfinite(
    targets
).all()

assert np.isfinite(
    affect_targets
).all()


# ============================================================
# Dataset statistics
# ============================================================

padding_positions = np.sum(
    mask == 0
)

trainable_targets = np.sum(
    target_mask == 1
)

print("\n--- Sequence Statistics ---")

print(
    f"Sequences:             {sequence_count:,}"
)

print(
    f"Real timesteps:        {real_timesteps:,}"
)

print(
    f"Trainable targets:     {trainable_targets:,}"
)

print(
    f"Padding positions:     {padding_positions:,}"
)

print(
    f"Padding percentage:    "
    f"{padding_positions / mask.size * 100:.2f}%"
)


# ============================================================
# Target statistics
# ============================================================

print("\n--- Correctness Target Statistics ---")

real_targets = targets[
    target_mask == 1
]

print(
    f"Min:    {real_targets.min()}"
)

print(
    f"Max:    {real_targets.max()}"
)

print(
    f"Mean:   {real_targets.mean():.6f}"
)

print(
    f"Unique values: {np.unique(real_targets)}"
)


print("\n--- Affect Target Statistics ---")

for i, affect_name in enumerate(
    affect_columns
):

    values = affect_targets[
        :, :, i
    ][
        target_mask == 1
    ]

    print(
        f"{affect_name}: "
        f"min={values.min():.6f}, "
        f"max={values.max():.6f}, "
        f"mean={values.mean():.6f}"
    )


# ============================================================
# Save
# ============================================================

print("\nSaving model sequences...")

np.savez_compressed(
    OUTPUT_FILE,

    X=X,
    mask=mask,

    targets=targets,
    affect_targets=affect_targets,
    target_mask=target_mask,

    problem_ids=problem_ids,
    skill_ids=skill_ids,

    user_ids=user_ids,
)

print(
    f"Saved to: {OUTPUT_FILE}"
)


# ============================================================
# Final success message
# ============================================================

print(
    "\nACDT TARGET + METADATA PREPARATION PASSED"
)