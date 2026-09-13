import os
import numpy as np


INPUT_FILE = "data/processed/model_sequences.npz"
OUTPUT_FILE = "data/processed/student_split.npz"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def main():

    print("Loading model sequences...")

    data = np.load(
        INPUT_FILE,
        allow_pickle=True
    )

    user_ids = data["user_ids"]

    
    
    

    unique_users = np.unique(user_ids)

    total_students = len(unique_users)

    print(f"Total unique students: {total_students}")

    
    
    

    if not np.isclose(
        TRAIN_RATIO + VAL_RATIO + TEST_RATIO,
        1.0
    ):
        raise ValueError(
            "Train/validation/test ratios must sum to 1."
        )

    
    
    

    rng = np.random.default_rng(RANDOM_SEED)

    shuffled_users = unique_users.copy()

    rng.shuffle(shuffled_users)

    
    
    

    train_end = int(
        total_students * TRAIN_RATIO
    )

    val_end = train_end + int(
        total_students * VAL_RATIO
    )

    
    
    

    train_users = shuffled_users[:train_end]

    val_users = shuffled_users[
        train_end:val_end
    ]

    test_users = shuffled_users[val_end:]

    
    
    

    print()
    print("========== STUDENT SPLIT ==========")

    print(
        f"Train students:      {len(train_users)} "
        f"({len(train_users) / total_students * 100:.2f}%)"
    )

    print(
        f"Validation students:  {len(val_users)} "
        f"({len(val_users) / total_students * 100:.2f}%)"
    )

    print(
        f"Test students:        {len(test_users)} "
        f"({len(test_users) / total_students * 100:.2f}%)"
    )

    print()

    
    
    

    train_set = set(train_users)
    val_set = set(val_users)
    test_set = set(test_users)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    print(
        f"Train/validation overlap: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train/test overlap: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation/test overlap: "
        f"{len(val_test_overlap)}"
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):
        raise ValueError(
            "ERROR: Student leakage detected!"
        )

    
    
    

    combined_users = np.concatenate([
        train_users,
        val_users,
        test_users
    ])

    unique_combined_users = np.unique(
        combined_users
    )

    print()

    print(
        f"Students represented in splits: "
        f"{len(unique_combined_users)}"
    )

    print(
        f"Students missing from splits: "
        f"{total_students - len(unique_combined_users)}"
    )

    if len(unique_combined_users) != total_students:
        raise ValueError(
            "ERROR: Some students are missing from the split!"
        )

    
    
    

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

    train_sequences = train_sequence_mask.sum()
    val_sequences = val_sequence_mask.sum()
    test_sequences = test_sequence_mask.sum()

    print()
    print("========== SEQUENCE DISTRIBUTION ==========")

    print(
        f"Train sequences:      {train_sequences}"
    )

    print(
        f"Validation sequences:  {val_sequences}"
    )

    print(
        f"Test sequences:        {test_sequences}"
    )

    print(
        f"Total sequences:       "
        f"{train_sequences + val_sequences + test_sequences}"
    )

    
    
    

    sequence_membership_count = (
        train_sequence_mask.astype(int)
        + val_sequence_mask.astype(int)
        + test_sequence_mask.astype(int)
    )

    invalid_sequence_assignments = np.sum(
        sequence_membership_count != 1
    )

    print(
        f"Invalid sequence assignments: "
        f"{invalid_sequence_assignments}"
    )

    if invalid_sequence_assignments != 0:
        raise ValueError(
            "ERROR: Some sequences do not belong to exactly one split!"
        )

    
    
    

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_FILE,
        train_users=train_users,
        val_users=val_users,
        test_users=test_users
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("Student-level split complete.")


if __name__ == "__main__":
    main()