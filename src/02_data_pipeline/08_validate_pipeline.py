import os
import numpy as np
import pandas as pd


RAW_FILE = "data/raw/2012-2013-data-with-predictions-4-final/2012-2013-data-with-predictions-4-final.csv"

CLEAN_FILE = "data/processed/clean_interactions.csv"

FEATURE_FILE = "data/processed/featured_interactions.csv"

AFFECT_FILE = "data/processed/affect_interactions.csv"

SEQUENCE_FILE = "data/processed/model_sequences.npz"

SPLIT_FILE = "data/processed/student_split.npz"


def check(condition, message):
    if not condition:
        raise ValueError(f"FAILED: {message}")

    print(f"PASS: {message}")


def main():

    print("========== ACDT PHASE 2 FINAL VALIDATION ==========")
    print()

    # ============================================================
    # Check pipeline files
    # ============================================================

    files = [
        RAW_FILE,
        CLEAN_FILE,
        FEATURE_FILE,
        AFFECT_FILE,
        SEQUENCE_FILE,
        SPLIT_FILE,
    ]

    print("Checking pipeline files...")

    for file in files:
        check(
            os.path.exists(file),
            f"File exists: {file}"
        )

    print()

    # ============================================================
    # Load model sequences
    # ============================================================

    print("Loading model sequences...")

    data = np.load(
        SEQUENCE_FILE,
        allow_pickle=True
    )

    X = data["X"]
    mask = data["mask"]

    targets = data["targets"]
    affect_targets = data["affect_targets"]
    target_mask = data["target_mask"]

    problem_ids = data["problem_ids"]
    skill_ids = data["skill_ids"]
    user_ids = data["user_ids"]

    print()

    # ============================================================
    # Shape validation
    # ============================================================

    check(
        X.ndim == 3,
        "X is 3-dimensional"
    )

    check(
        X.shape[1] == 50,
        "Sequence length is 50"
    )

    check(
        X.shape[2] == 11,
        "Feature count is 11"
    )

    check(
        mask.shape == X.shape[:2],
        "Mask shape matches X"
    )

    check(
        targets.shape == mask.shape,
        "Targets shape matches mask"
    )

    check(
        affect_targets.shape[:2] == mask.shape,
        "Affect targets sequence shape matches mask"
    )

    check(
        affect_targets.shape[2] == 4,
        "Affect targets contain 4 affect dimensions"
    )

    check(
        target_mask.shape == mask.shape,
        "Target mask shape matches mask"
    )

    check(
        problem_ids.shape == mask.shape,
        "Problem ID shape matches mask"
    )

    check(
        skill_ids.shape == mask.shape,
        "Skill ID shape matches mask"
    )

    check(
        len(user_ids) == len(X),
        "One user ID exists for every sequence"
    )

    print()

    # ============================================================
    # Load student splits
    # ============================================================

    print("Loading student splits...")

    split_data = np.load(
        SPLIT_FILE,
        allow_pickle=True
    )

    train_users = split_data["train_users"]
    val_users = split_data["val_users"]
    test_users = split_data["test_users"]

    train_set = set(train_users)
    val_set = set(val_users)
    test_set = set(test_users)

    check(
        len(train_set & val_set) == 0,
        "No train/validation student overlap"
    )

    check(
        len(train_set & test_set) == 0,
        "No train/test student overlap"
    )

    check(
        len(val_set & test_set) == 0,
        "No validation/test student overlap"
    )

    split_users = np.concatenate([
        train_users,
        val_users,
        test_users
    ])

    check(
        len(np.unique(split_users)) == len(split_users),
        "Every student appears in exactly one split"
    )

    print()

    # ============================================================
    # Validate sequence split membership
    # ============================================================

    train_sequence_mask = np.isin(
        user_ids,
        train_users
    )

    val_sequence_mask = np.isin(
        user_ids,
        val_users
    )

    test_sequence_mask = np.isin(
        user_ids,
        test_users
    )

    sequence_membership = (
        train_sequence_mask.astype(int)
        + val_sequence_mask.astype(int)
        + test_sequence_mask.astype(int)
    )

    check(
        np.all(sequence_membership == 1),
        "Every sequence belongs to exactly one split"
    )

    print()

    # ============================================================
    # Padding validation
    # ============================================================

    padding_positions = mask == 0
    real_positions = mask == 1

    check(
        np.all(
            target_mask[padding_positions] == 0
        ),
        "Padding positions have no training targets"
    )

    check(
        np.all(target_mask <= mask),
        "Target mask never exceeds sequence mask"
    )

    check(
        np.all(
            X[padding_positions] == 0
        ),
        "Padding feature values are zero"
    )

    check(
        np.all(
            targets[padding_positions] == 0
        ),
        "Padding correctness targets are zero"
    )

    check(
        np.all(
            affect_targets[target_mask == 0] == 0
        ),
        "Invalid positions have no affect targets"
    )

    check(
        np.all(
            problem_ids[padding_positions] == ""
        ),
        "Padding problem IDs are empty"
    )

    check(
        np.all(
            skill_ids[padding_positions] == -1
        ),
        "Padding skill IDs are -1"
    )

    print()

    # ============================================================
    # Target validation
    # ============================================================

    target_positions = target_mask == 1

    check(
        np.all(
            np.isfinite(
                targets[target_positions]
            )
        ),
        "All training targets are finite"
    )

    check(
        np.all(
            np.isfinite(
                affect_targets[target_positions]
            )
        ),
        "All affect targets are finite"
    )

    # ============================================================
    # Correctness target values
    # ============================================================

    observed_targets = np.unique(
        targets[target_positions]
    )

    print()

    print("Observed target correctness values:")

    print(observed_targets)

    print(
        "Number of unique target values:",
        len(observed_targets)
    )

    check(
        np.all(
            np.isfinite(observed_targets)
        ),
        "Target correctness values are finite"
    )

    # ============================================================
    # Affect target validation
    # ============================================================

    observed_affect_targets = affect_targets[
        target_positions
    ]

    check(
        np.all(
            observed_affect_targets >= 0
        ),
        "Affect target values are >= 0"
    )

    check(
        np.all(
            observed_affect_targets <= 1
        ),
        "Affect target values are <= 1"
    )

    print()

    print("Observed affect target ranges:")

    affect_names = [
        "Frustrated",
        "Confused",
        "Concentrating",
        "Bored"
    ]

    for i, name in enumerate(affect_names):

        values = observed_affect_targets[:, i]

        print(
            f"{name}: "
            f"min={values.min():.6f}, "
            f"max={values.max():.6f}, "
            f"mean={values.mean():.6f}"
        )

    # ============================================================
    # Global finite-value validation
    # ============================================================

    check(
        np.all(
            np.isfinite(X)
        ),
        "X contains no NaN or infinite values"
    )

    check(
        np.all(
            np.isfinite(targets)
        ),
        "Targets contain no NaN or infinite values"
    )

    check(
        np.all(
            np.isfinite(affect_targets)
        ),
        "Affect targets contain no NaN or infinite values"
    )

    print()

    # ============================================================
    # Dataset coverage
    # ============================================================

    real_timesteps = int(
        mask.sum()
    )

    print(
        f"Real interaction timesteps: {real_timesteps:,}"
    )

    check(
        real_timesteps == 6_123_270,
        "All 6,123,270 original interactions are represented"
    )

    print()

    # ============================================================
    # Student coverage
    # ============================================================

    sequence_users = np.unique(
        user_ids
    )

    split_users_unique = np.unique(
        split_users
    )

    check(
        len(sequence_users) == 46_674,
        "All 46,674 students appear in sequences"
    )

    check(
        np.array_equal(
            np.sort(sequence_users),
            np.sort(split_users_unique)
        ),
        "Sequence students exactly match split students"
    )

    print()

    # ============================================================
    # Final summary
    # ============================================================

    print("========== FINAL SUMMARY ==========")

    print(
        f"Students:             {len(sequence_users):,}"
    )

    print(
        f"Sequences:            {len(X):,}"
    )

    print(
        f"Sequence shape:       {X.shape}"
    )

    print(
        f"Real interactions:    {real_timesteps:,}"
    )

    print(
        f"Train targets:        {target_mask.sum():,}"
    )

    print(
        f"Affect targets:       {affect_targets.shape}"
    )

    print(
        f"Padding positions:    {padding_positions.sum():,}"
    )

    print(
        f"Train students:       {len(train_users):,}"
    )

    print(
        f"Validation students:  {len(val_users):,}"
    )

    print(
        f"Test students:        {len(test_users):,}"
    )

    print()

    print("===================================")
    print("ACDT PHASE 2 VALIDATION PASSED")
    print("Data pipeline is model-ready.")
    print("===================================")


if __name__ == "__main__":
    main()