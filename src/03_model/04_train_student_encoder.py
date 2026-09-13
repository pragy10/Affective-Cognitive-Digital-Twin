import os
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

MODEL_INPUT_FILE = "data/processed/model_inputs.npz"
CATEGORICAL_INPUT_FILE = "data/processed/categorical_inputs.npz"
MODEL_SEQUENCE_FILE = "data/processed/model_sequences.npz"
VOCAB_FILE = "data/processed/embedding_vocab.npz"
SPLIT_FILE = "data/processed/student_split.npz"

CHECKPOINT_DIR = "models"
BEST_MODEL_FILE = os.path.join(
    CHECKPOINT_DIR,
    "student_encoder_best.pt"
)

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

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
# Import StudentEncoder
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
# Masked Huber Loss
# ============================================================

def masked_huber_loss(
    predictions,
    targets,
    mask
):

    loss_function = nn.HuberLoss(
        reduction="none"
    )

    loss = loss_function(
        predictions,
        targets
    )

    if mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)

    mask = mask.expand_as(loss)

    masked_loss = loss * mask

    denominator = mask.sum().clamp(
        min=1.0
    )

    return masked_loss.sum() / denominator


# ============================================================
# Training function
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_cognitive_loss = 0.0
    total_affective_loss = 0.0

    batches = 0

    for batch in dataloader:

        X = batch["X"].to(device)

        problem_indices = batch[
            "problem_indices"
        ].to(device)

        skill_indices = batch[
            "skill_indices"
        ].to(device)

        skill_missing = batch[
            "skill_missing"
        ].to(device)

        mask = batch[
            "mask"
        ].to(device)

        targets = batch[
            "targets"
        ].to(device)

        affect_targets = batch[
            "affect_targets"
        ].to(device)

        target_mask = batch[
            "target_mask"
        ].to(device)

        # ----------------------------------------------------
        # Transformer padding mask
        #
        # model expects:
        # True  = padding
        # False = real timestep
        # ----------------------------------------------------

        padding_mask = ~mask.bool()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            numeric_features=X,
            problem_indices=problem_indices,
            skill_indices=skill_indices,
            skill_missing=skill_missing,
            padding_mask=padding_mask
        )

        next_correctness = outputs[
            "next_correctness"
        ]

        next_affect = outputs[
            "next_affect"
        ]

        # ----------------------------------------------------
        # Cognitive loss
        # ----------------------------------------------------

        cognitive_loss = masked_huber_loss(
            next_correctness,
            targets,
            target_mask
        )

        # ----------------------------------------------------
        # Affective loss
        # ----------------------------------------------------

        affective_loss = masked_huber_loss(
            next_affect,
            affect_targets,
            target_mask
        )

        # ----------------------------------------------------
        # Joint loss
        # ----------------------------------------------------

        loss = (
            cognitive_loss
            + affective_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        total_cognitive_loss += (
            cognitive_loss.item()
        )

        total_affective_loss += (
            affective_loss.item()
        )

        batches += 1

    return (
        total_loss / batches,
        total_cognitive_loss / batches,
        total_affective_loss / batches
    )


# ============================================================
# Validation function
# ============================================================

@torch.no_grad()
def validate(
    model,
    dataloader
):

    model.eval()

    total_loss = 0.0
    total_cognitive_loss = 0.0
    total_affective_loss = 0.0

    batches = 0

    for batch in dataloader:

        X = batch["X"].to(device)

        problem_indices = batch[
            "problem_indices"
        ].to(device)

        skill_indices = batch[
            "skill_indices"
        ].to(device)

        skill_missing = batch[
            "skill_missing"
        ].to(device)

        mask = batch[
            "mask"
        ].to(device)

        targets = batch[
            "targets"
        ].to(device)

        affect_targets = batch[
            "affect_targets"
        ].to(device)

        target_mask = batch[
            "target_mask"
        ].to(device)

        padding_mask = ~mask.bool()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            numeric_features=X,
            problem_indices=problem_indices,
            skill_indices=skill_indices,
            skill_missing=skill_missing,
            padding_mask=padding_mask
        )

        next_correctness = outputs[
            "next_correctness"
        ]

        next_affect = outputs[
            "next_affect"
        ]

        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------

        cognitive_loss = masked_huber_loss(
            next_correctness,
            targets,
            target_mask
        )

        affective_loss = masked_huber_loss(
            next_affect,
            affect_targets,
            target_mask
        )

        loss = (
            cognitive_loss
            + affective_loss
        )

        total_loss += loss.item()

        total_cognitive_loss += (
            cognitive_loss.item()
        )

        total_affective_loss += (
            affective_loss.item()
        )

        batches += 1

    return (
        total_loss / batches,
        total_cognitive_loss / batches,
        total_affective_loss / batches
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "========== ACDT STUDENT ENCODER TRAINING =========="
    )

    # ========================================================
    # Load processed inputs
    # ========================================================

    print("\nLoading processed inputs...")

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

    # ========================================================
    # Load arrays
    # ========================================================

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

    mask = model_sequences[
        "mask"
    ]

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

    # ========================================================
    # Student splits
    # ========================================================

    train_users = split_data[
        "train_users"
    ]

    val_users = split_data[
        "val_users"
    ]

    # ========================================================
    # Basic validation
    # ========================================================

    print("\nChecking loaded data...")

    print(
        f"X shape:              {X.shape}"
    )

    print(
        f"Problem indices:      {problem_indices.shape}"
    )

    print(
        f"Skill indices:        {skill_indices.shape}"
    )

    print(
        f"Skill missing:        {skill_missing.shape}"
    )

    print(
        f"Mask:                 {mask.shape}"
    )

    print(
        f"Targets:              {targets.shape}"
    )

    print(
        f"Affect targets:       {affect_targets.shape}"
    )

    print(
        f"Target mask:          {target_mask.shape}"
    )

    assert X.shape[:2] == problem_indices.shape
    assert X.shape[:2] == skill_indices.shape
    assert X.shape[:2] == skill_missing.shape
    assert X.shape[:2] == mask.shape
    assert X.shape[:2] == targets.shape
    assert X.shape[:2] == affect_targets.shape[:2]
    assert affect_targets.shape[2] == 4
    assert X.shape[:2] == target_mask.shape

    print(
        "All input shapes validated."
    )

    # ========================================================
    # Build train / validation sequence masks
    # ========================================================

    train_sequence_mask = np.isin(
        user_ids,
        train_users
    )

    val_sequence_mask = np.isin(
        user_ids,
        val_users
    )

    train_count = int(
        train_sequence_mask.sum()
    )

    val_count = int(
        val_sequence_mask.sum()
    )

    print(
        f"\nTraining sequences:   {train_count:,}"
    )

    print(
        f"Validation sequences: {val_count:,}"
    )

    # ========================================================
    # Create datasets
    # ========================================================

    train_dataset = StudentSequenceDataset(
        X=X[train_sequence_mask],
        problem_indices=problem_indices[
            train_sequence_mask
        ],
        skill_indices=skill_indices[
            train_sequence_mask
        ],
        skill_missing=skill_missing[
            train_sequence_mask
        ],
        mask=mask[
            train_sequence_mask
        ],
        targets=targets[
            train_sequence_mask
        ],
        affect_targets=affect_targets[
            train_sequence_mask
        ],
        target_mask=target_mask[
            train_sequence_mask
        ]
    )

    val_dataset = StudentSequenceDataset(
        X=X[val_sequence_mask],
        problem_indices=problem_indices[
            val_sequence_mask
        ],
        skill_indices=skill_indices[
            val_sequence_mask
        ],
        skill_missing=skill_missing[
            val_sequence_mask
        ],
        mask=mask[
            val_sequence_mask
        ],
        targets=targets[
            val_sequence_mask
        ],
        affect_targets=affect_targets[
            val_sequence_mask
        ],
        target_mask=target_mask[
            val_sequence_mask
        ]
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    print(
        f"\nTraining batches:    {len(train_loader):,}"
    )

    print(
        f"Validation batches:  {len(val_loader):,}"
    )

    # ========================================================
    # Vocabulary sizes
    # ========================================================

    # embedding_vocab contains actual IDs only.
    # Index 0 is reserved for padding/unknown.
    num_problems = (
        len(vocab["problem_ids"]) + 1
    )

    num_skills = (
        len(vocab["skill_ids"]) + 1
    )

    print(
        f"\nProblem vocabulary:  {num_problems:,}"
    )

    print(
        f"Skill vocabulary:    {num_skills:,}"
    )

    # ========================================================
    # Create model
    # ========================================================

    print("\nCreating Student Encoder...")

    # IMPORTANT:
    # StudentEncoder's actual constructor is:
    #
    # StudentEncoder(
    #     num_problems,
    #     num_skills
    # )
    #
    # The architecture dimensions are already defined
    # inside 03_student_encoder.py.

    model = StudentEncoder(
        num_problems=num_problems,
        num_skills=num_skills
    ).to(device)

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # Parameter count
    # ========================================================

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters:     {parameter_count:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameter_count:,}"
    )

    # ========================================================
    # Checkpoint directory
    # ========================================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # ========================================================
    # Training loop
    # ========================================================

    best_validation_loss = float("inf")

    print()
    print(
        "===================================================="
    )

    print(
        "Starting training..."
    )

    print(
        "===================================================="
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"\n---------- Epoch {epoch}/{EPOCHS} ----------"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        (
            train_loss,
            train_cognitive,
            train_affective
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        print(
            f"Train total loss:      {train_loss:.6f}"
        )

        print(
            f"Train cognitive loss:  {train_cognitive:.6f}"
        )

        print(
            f"Train affective loss:  {train_affective:.6f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_cognitive,
            val_affective
        ) = validate(
            model,
            val_loader
        )

        print(
            f"Val total loss:        {val_loss:.6f}"
        )

        print(
            f"Val cognitive loss:    {val_cognitive:.6f}"
        )

        print(
            f"Val affective loss:    {val_affective:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_validation_loss:

            best_validation_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "validation_loss":
                        val_loss,

                    "validation_cognitive_loss":
                        val_cognitive,

                    "validation_affective_loss":
                        val_affective,

                    "config": {
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "gradient_clip": GRAD_CLIP,
                        "sequence_length": 50,
                        "numeric_features": 14,
                        "problem_embedding_dim": 32,
                        "skill_embedding_dim": 16,
                        "model_dim": 128,
                        "num_heads": 4,
                        "num_layers": 2,
                        "ffn_dim": 256,
                        "dropout": 0.1,
                        "cognitive_dim": 64,
                        "affective_dim": 64
                    }
                },
                BEST_MODEL_FILE
            )

            print(
                f"Best model saved to: {BEST_MODEL_FILE}"
            )

    # ========================================================
    # Final result
    # ========================================================

    print()

    print(
        "===================================================="
    )

    print(
        "ACDT STUDENT ENCODER TRAINING COMPLETE"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Best checkpoint: {BEST_MODEL_FILE}"
    )

    print(
        "===================================================="
    )


if __name__ == "__main__":
    main()