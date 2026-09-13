import os
import importlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

MODEL_INPUT_FILE = "data/processed/model_inputs.npz"
CATEGORICAL_INPUT_FILE = "data/processed/categorical_inputs.npz"
MODEL_SEQUENCE_FILE = "data/processed/model_sequences.npz"
VOCAB_FILE = "data/processed/embedding_vocab.npz"
SPLIT_FILE = "data/processed/student_split.npz"

CHECKPOINT_FILE = "models/student_encoder_best.pt"

BATCH_SIZE = 32
NUM_WORKERS = 0


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if device.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# Import model
# ============================================================

student_encoder_module = importlib.import_module(
    "src.03_model.03_student_encoder"
)

StudentEncoder = (
    student_encoder_module.StudentEncoder
)


# ============================================================
# Dataset
# ============================================================

class StudentSequenceDataset(Dataset):

    def __init__(
        self,
        X,
        problem_indices,
        skill_indices,
        skill_missing,
        mask,
        targets,
        affect_targets,
        target_mask
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.problem_indices = torch.tensor(
            problem_indices,
            dtype=torch.long
        )

        self.skill_indices = torch.tensor(
            skill_indices,
            dtype=torch.long
        )

        self.skill_missing = torch.tensor(
            skill_missing,
            dtype=torch.float32
        )

        self.mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.float32
        )

        self.affect_targets = torch.tensor(
            affect_targets,
            dtype=torch.float32
        )

        self.target_mask = torch.tensor(
            target_mask,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        return {
            "X": self.X[index],
            "problem_indices": self.problem_indices[index],
            "skill_indices": self.skill_indices[index],
            "skill_missing": self.skill_missing[index],
            "mask": self.mask[index],
            "targets": self.targets[index],
            "affect_targets": self.affect_targets[index],
            "target_mask": self.target_mask[index],
        }


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():

    print()
    print(
        "========== ACDT STUDENT ENCODER EVALUATION =========="
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading processed data...")

    model_inputs = np.load(
        MODEL_INPUT_FILE,
        allow_pickle=True
    )

    categorical_inputs = np.load(
        CATEGORICAL_INPUT_FILE,
        allow_pickle=True
    )

    model_sequences = np.load(
        MODEL_SEQUENCE_FILE,
        allow_pickle=True
    )

    vocab = np.load(
        VOCAB_FILE,
        allow_pickle=True
    )

    split_data = np.load(
        SPLIT_FILE,
        allow_pickle=True
    )

    X = model_inputs["X"]

    problem_indices = categorical_inputs[
        "problem_indices"
    ]

    skill_indices = categorical_inputs[
        "skill_indices"
    ]

    skill_missing = categorical_inputs[
        "skill_missing"
    ]

    mask = model_sequences["mask"]

    targets = model_sequences["targets"]

    affect_targets = model_sequences[
        "affect_targets"
    ]

    target_mask = model_sequences[
        "target_mask"
    ]

    user_ids = model_sequences[
        "user_ids"
    ]

    test_users = split_data[
        "test_users"
    ]

    # --------------------------------------------------------
    # Select TEST students only
    # --------------------------------------------------------

    test_sequence_mask = np.isin(
        user_ids,
        test_users
    )

    test_count = int(
        test_sequence_mask.sum()
    )

    print(
        f"\nTest students:       {len(test_users):,}"
    )

    print(
        f"Test sequences:      {test_count:,}"
    )

    test_dataset = StudentSequenceDataset(
        X=X[test_sequence_mask],
        problem_indices=problem_indices[
            test_sequence_mask
        ],
        skill_indices=skill_indices[
            test_sequence_mask
        ],
        skill_missing=skill_missing[
            test_sequence_mask
        ],
        mask=mask[
            test_sequence_mask
        ],
        targets=targets[
            test_sequence_mask
        ],
        affect_targets=affect_targets[
            test_sequence_mask
        ],
        target_mask=target_mask[
            test_sequence_mask
        ]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    num_problems = (
        len(vocab["problem_ids"]) + 1
    )

    num_skills = (
        len(vocab["skill_ids"]) + 1
    )

    model = StudentEncoder(
        num_problems=num_problems,
        num_skills=num_skills
    ).to(device)

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print(
        f"\nLoading checkpoint: {CHECKPOINT_FILE}"
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: {checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.6f}"
    )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    cognitive_absolute_error = 0.0
    cognitive_squared_error = 0.0

    affect_absolute_error = np.zeros(4)
    affect_squared_error = np.zeros(4)

    total_cognitive_values = 0
    total_affect_values = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nEvaluating test students...")

    for batch in test_loader:

        X_batch = batch["X"].to(device)

        problem_indices_batch = batch[
            "problem_indices"
        ].to(device)

        skill_indices_batch = batch[
            "skill_indices"
        ].to(device)

        skill_missing_batch = batch[
            "skill_missing"
        ].to(device)

        mask_batch = batch[
            "mask"
        ].to(device)

        targets_batch = batch[
            "targets"
        ].to(device)

        affect_targets_batch = batch[
            "affect_targets"
        ].to(device)

        target_mask_batch = batch[
            "target_mask"
        ].to(device)

        padding_mask = ~mask_batch.bool()

        outputs = model(
            numeric_features=X_batch,
            problem_indices=problem_indices_batch,
            skill_indices=skill_indices_batch,
            skill_missing=skill_missing_batch,
            padding_mask=padding_mask
        )

        predicted_correctness = outputs[
            "next_correctness"
        ]

        predicted_affect = outputs[
            "next_affect"
        ]

        # ----------------------------------------------------
        # Cognitive metrics
        # ----------------------------------------------------

        valid_cognitive = (
            target_mask_batch.bool()
        )

        cognitive_errors = (
            predicted_correctness[
                valid_cognitive
            ]
            -
            targets_batch[
                valid_cognitive
            ]
        )

        cognitive_absolute_error += (
            torch.abs(
                cognitive_errors
            ).sum().item()
        )

        cognitive_squared_error += (
            torch.square(
                cognitive_errors
            ).sum().item()
        )

        total_cognitive_values += (
            cognitive_errors.numel()
        )

        # ----------------------------------------------------
        # Affect metrics
        # ----------------------------------------------------

        valid_affect = (
            target_mask_batch.bool()
            .unsqueeze(-1)
            .expand_as(
                affect_targets_batch
            )
        )

        affect_errors = (
            predicted_affect[
                valid_affect
            ]
            -
            affect_targets_batch[
                valid_affect
            ]
        )

        affect_errors = affect_errors.reshape(
            -1,
            4
        )

        # MAE/RMSE will be accumulated
        # per affect dimension.

        affect_absolute_error += (
            torch.abs(
                affect_errors
            ).sum(dim=0)
            .cpu()
            .numpy()
        )

        affect_squared_error += (
            torch.square(
                affect_errors
            ).sum(dim=0)
            .cpu()
            .numpy()
        )

        total_affect_values += (
            affect_errors.shape[0]
        )

    # ========================================================
    # Final metrics
    # ========================================================

    cognitive_mae = (
        cognitive_absolute_error
        /
        total_cognitive_values
    )

    cognitive_rmse = np.sqrt(
        cognitive_squared_error
        /
        total_cognitive_values
    )

    affect_mae = (
        affect_absolute_error
        /
        total_affect_values
    )

    affect_rmse = np.sqrt(
        affect_squared_error
        /
        total_affect_values
    )

    overall_affect_mae = (
        affect_absolute_error.sum()
        /
        (total_affect_values * 4)
    )

    overall_affect_rmse = np.sqrt(
        affect_squared_error.sum()
        /
        (total_affect_values * 4)
    )

    # ========================================================
    # Print results
    # ========================================================

    print()
    print(
        "===================================================="
    )

    print(
        "ACDT STUDENT ENCODER TEST RESULTS"
    )

    print(
        "===================================================="
    )

    print(
        f"\nCognitive prediction:"
    )

    print(
        f"MAE:  {cognitive_mae:.6f}"
    )

    print(
        f"RMSE: {cognitive_rmse:.6f}"
    )

    print(
        f"\nAffective prediction:"
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
            f"{name:<15} "
            f"MAE: {affect_mae[i]:.6f}  "
            f"RMSE: {affect_rmse[i]:.6f}"
        )

    print(
        f"\nOverall affect MAE:  "
        f"{overall_affect_mae:.6f}"
    )

    print(
        f"Overall affect RMSE: "
        f"{overall_affect_rmse:.6f}"
    )

    print(
        f"\nEvaluated cognitive targets: "
        f"{total_cognitive_values:,}"
    )

    print(
        f"Evaluated affect targets: "
        f"{total_affect_values:,}"
    )

    print(
        "\n===================================================="
    )

    print(
        "ACDT STUDENT ENCODER EVALUATION COMPLETE"
    )

    print(
        "===================================================="
    )


if __name__ == "__main__":
    main()