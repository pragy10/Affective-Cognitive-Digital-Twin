import numpy as np


# ============================================================
# Configuration
# ============================================================

MODEL_SEQUENCE_FILE = "data/processed/model_sequences.npz"
SPLIT_FILE = "data/processed/student_split.npz"


# ============================================================
# Metrics
# ============================================================

def calculate_mae(predictions, targets):
    return np.mean(
        np.abs(predictions - targets)
    )


def calculate_rmse(predictions, targets):
    return np.sqrt(
        np.mean(
            np.square(predictions - targets)
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "========== ACDT BASELINE EVALUATION =========="
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading processed data...")

    model_sequences = np.load(
        MODEL_SEQUENCE_FILE,
        allow_pickle=True
    )

    split_data = np.load(
        SPLIT_FILE,
        allow_pickle=True
    )

    targets = model_sequences[
        "targets"
    ]

    affect_targets = model_sequences[
        "affect_targets"
    ]

    target_mask = model_sequences[
        "target_mask"
    ]

    user_ids = model_sequences[
        "user_ids"
    ]

    train_users = split_data[
        "train_users"
    ]

    test_users = split_data[
        "test_users"
    ]

    # --------------------------------------------------------
    # Identify train/test sequences
    # --------------------------------------------------------

    train_sequence_mask = np.isin(
        user_ids,
        train_users
    )

    test_sequence_mask = np.isin(
        user_ids,
        test_users
    )

    print(
        f"\nTraining sequences: "
        f"{train_sequence_mask.sum():,}"
    )

    print(
        f"Test sequences:     "
        f"{test_sequence_mask.sum():,}"
    )

    # --------------------------------------------------------
    # Extract valid training targets
    # --------------------------------------------------------

    train_target_mask = target_mask[
        train_sequence_mask
    ].astype(bool)

    train_targets = targets[
        train_sequence_mask
    ][train_target_mask]

    train_affect_mask = (
        target_mask[
            train_sequence_mask
        ]
        .astype(bool)
        .reshape(-1)
    )

    train_affect_targets = affect_targets[
        train_sequence_mask
    ].reshape(-1, 4)[train_affect_mask]

    # --------------------------------------------------------
    # Calculate training-set means
    # --------------------------------------------------------

    cognitive_mean = np.mean(
        train_targets
    )

    affect_means = np.mean(
        train_affect_targets,
        axis=0
    )

    print(
        "\nTraining-set baseline values:"
    )

    print(
        f"Cognitive mean:       "
        f"{cognitive_mean:.6f}"
    )

    affect_names = [
        "Frustration",
        "Confusion",
        "Concentration",
        "Boredom"
    ]

    for i, name in enumerate(
        affect_names
    ):

        print(
            f"{name:<20}"
            f"{affect_means[i]:.6f}"
        )

    # --------------------------------------------------------
    # Extract valid test targets
    # --------------------------------------------------------

    test_target_mask = target_mask[
        test_sequence_mask
    ].astype(bool)

    test_targets = targets[
        test_sequence_mask
    ][test_target_mask]

    test_affect_mask = (
        target_mask[
            test_sequence_mask
        ]
        .astype(bool)
        .reshape(-1)
    )

    test_affect_targets = affect_targets[
        test_sequence_mask
    ].reshape(-1, 4)[test_affect_mask]

    # --------------------------------------------------------
    # Cognitive baseline
    # --------------------------------------------------------

    cognitive_predictions = np.full(
        test_targets.shape,
        cognitive_mean,
        dtype=np.float32
    )

    cognitive_mae = calculate_mae(
        cognitive_predictions,
        test_targets
    )

    cognitive_rmse = calculate_rmse(
        cognitive_predictions,
        test_targets
    )

    # --------------------------------------------------------
    # Affective baseline
    # --------------------------------------------------------

    affect_predictions = np.tile(
        affect_means,
        (
            test_affect_targets.shape[0],
            1
        )
    )

    affect_mae = np.mean(
        np.abs(
            affect_predictions
            -
            test_affect_targets
        ),
        axis=0
    )

    affect_rmse = np.sqrt(
        np.mean(
            np.square(
                affect_predictions
                -
                test_affect_targets
            ),
            axis=0
        )
    )

    overall_affect_mae = np.mean(
        affect_mae
    )

    overall_affect_rmse = np.mean(
        affect_rmse
    )

    # --------------------------------------------------------
    # Compare with Student Encoder
    # --------------------------------------------------------

    encoder_cognitive_mae = 0.344338
    encoder_cognitive_rmse = 0.420084

    encoder_affect_mae = np.array([
        0.050127,
        0.079557,
        0.068345,
        0.221871
    ])

    encoder_affect_rmse = np.array([
        0.102807,
        0.189389,
        0.117846,
        0.274003
    ])

    encoder_overall_affect_mae = 0.104975
    encoder_overall_affect_rmse = 0.183985

    # --------------------------------------------------------
    # Improvement calculations
    # --------------------------------------------------------

    cognitive_mae_improvement = (
        (cognitive_mae - encoder_cognitive_mae)
        / cognitive_mae
        * 100
    )

    cognitive_rmse_improvement = (
        (cognitive_rmse - encoder_cognitive_rmse)
        / cognitive_rmse
        * 100
    )

    affect_mae_improvement = (
        (affect_mae - encoder_affect_mae)
        / affect_mae
        * 100
    )

    affect_rmse_improvement = (
        (affect_rmse - encoder_affect_rmse)
        / affect_rmse
        * 100
    )

    overall_affect_mae_improvement = (
        (
            overall_affect_mae
            -
            encoder_overall_affect_mae
        )
        /
        overall_affect_mae
        * 100
    )

    overall_affect_rmse_improvement = (
        (
            overall_affect_rmse
            -
            encoder_overall_affect_rmse
        )
        /
        overall_affect_rmse
        * 100
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print(
        "===================================================="
    )

    print(
        "ACDT BASELINE RESULTS"
    )

    print(
        "===================================================="
    )

    print(
        "\nCognitive prediction:"
    )

    print(
        f"Baseline MAE:       "
        f"{cognitive_mae:.6f}"
    )

    print(
        f"Student Encoder MAE:"
        f" {encoder_cognitive_mae:.6f}"
    )

    print(
        f"Improvement:         "
        f"{cognitive_mae_improvement:.2f}%"
    )

    print()

    print(
        f"Baseline RMSE:       "
        f"{cognitive_rmse:.6f}"
    )

    print(
        f"Student Encoder RMSE:"
        f" {encoder_cognitive_rmse:.6f}"
    )

    print(
        f"Improvement:         "
        f"{cognitive_rmse_improvement:.2f}%"
    )

    print(
        "\nAffective prediction:"
    )

    for i, name in enumerate(
        affect_names
    ):

        print(
            f"\n{name}"
        )

        print(
            f"Baseline MAE:        "
            f"{affect_mae[i]:.6f}"
        )

        print(
            f"Student Encoder MAE: "
            f"{encoder_affect_mae[i]:.6f}"
        )

        print(
            f"Improvement:         "
            f"{affect_mae_improvement[i]:.2f}%"
        )

        print(
            f"Baseline RMSE:       "
            f"{affect_rmse[i]:.6f}"
        )

        print(
            f"Student Encoder RMSE:"
            f" {encoder_affect_rmse[i]:.6f}"
        )

        print(
            f"Improvement:         "
            f"{affect_rmse_improvement[i]:.2f}%"
        )

    print(
        "\nOverall affect:"
    )

    print(
        f"Baseline MAE:        "
        f"{overall_affect_mae:.6f}"
    )

    print(
        f"Student Encoder MAE: "
        f"{encoder_overall_affect_mae:.6f}"
    )

    print(
        f"Improvement:         "
        f"{overall_affect_mae_improvement:.2f}%"
    )

    print()

    print(
        f"Baseline RMSE:       "
        f"{overall_affect_rmse:.6f}"
    )

    print(
        f"Student Encoder RMSE:"
        f" {encoder_overall_affect_rmse:.6f}"
    )

    print(
        f"Improvement:         "
        f"{overall_affect_rmse_improvement:.2f}%"
    )

    print(
        "\nTest cognitive targets: "
        f"{len(test_targets):,}"
    )

    print(
        "Test affect targets:    "
        f"{len(test_affect_targets):,}"
    )

    print()
    print(
        "===================================================="
    )

    print(
        "ACDT BASELINE EVALUATION COMPLETE"
    )

    print(
        "===================================================="
    )


if __name__ == "__main__":
    main()