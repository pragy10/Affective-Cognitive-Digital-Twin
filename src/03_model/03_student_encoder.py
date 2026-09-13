import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 50

NUMERIC_FEATURES = 14

PROBLEM_EMBED_DIM = 32
SKILL_EMBED_DIM = 16

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 256
DROPOUT = 0.1

COGNITIVE_DIM = 64
AFFECTIVE_DIM = 64

BATCH_SIZE = 32


# ============================================================
# Student Encoder
# ============================================================

class StudentEncoder(nn.Module):

    def __init__(
        self,
        num_problems,
        num_skills
    ):
        super().__init__()

        # ----------------------------------------------------
        # Categorical embeddings
        # ----------------------------------------------------

        self.problem_embedding = nn.Embedding(
            num_embeddings=num_problems,
            embedding_dim=PROBLEM_EMBED_DIM,
            padding_idx=0
        )

        self.skill_embedding = nn.Embedding(
            num_embeddings=num_skills,
            embedding_dim=SKILL_EMBED_DIM,
            padding_idx=0
        )

        # ----------------------------------------------------
        # Input projection
        #
        # 14 numeric
        # + 32 problem
        # + 16 skill
        # + 1 skill-missing
        # = 63
        # ----------------------------------------------------

        input_dim = (
            NUMERIC_FEATURES
            + PROBLEM_EMBED_DIM
            + SKILL_EMBED_DIM
            + 1
        )

        self.input_projection = nn.Linear(
            input_dim,
            D_MODEL
        )

        # ----------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            SEQUENCE_LENGTH,
            D_MODEL
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=NUM_LAYERS
        )

        # ----------------------------------------------------
        # Cognitive / affective state projections
        # ----------------------------------------------------

        self.cognitive_projection = nn.Linear(
            D_MODEL,
            COGNITIVE_DIM
        )

        self.affective_projection = nn.Linear(
            D_MODEL,
            AFFECTIVE_DIM
        )

        # ----------------------------------------------------
        # Prediction heads
        # ----------------------------------------------------

        self.correctness_head = nn.Linear(
            COGNITIVE_DIM,
            1
        )

        self.affect_head = nn.Linear(
            AFFECTIVE_DIM,
            4
        )

    def create_causal_mask(self, device):
        """
        Prevent timestep t from attending to future timesteps.
        """

        mask = torch.triu(
            torch.ones(
                SEQUENCE_LENGTH,
                SEQUENCE_LENGTH,
                device=device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        return mask

    def forward(
        self,
        numeric_features,
        problem_indices,
        skill_indices,
        skill_missing,
        padding_mask
    ):

        # ----------------------------------------------------
        # 1. Convert problem / skill IDs to embeddings
        # ----------------------------------------------------

        problem_emb = self.problem_embedding(
            problem_indices
        )

        skill_emb = self.skill_embedding(
            skill_indices
        )

        # ----------------------------------------------------
        # 2. Combine all timestep features
        # ----------------------------------------------------

        x = torch.cat(
            [
                numeric_features,
                problem_emb,
                skill_emb,
                skill_missing.unsqueeze(-1)
            ],
            dim=-1
        )

        # Expected:
        # (batch, 50, 63)

        # ----------------------------------------------------
        # 3. Project 63 → 128
        # ----------------------------------------------------

        x = self.input_projection(x)

        # Expected:
        # (batch, 50, 128)

        # ----------------------------------------------------
        # 4. Add positional information
        # ----------------------------------------------------

        positions = torch.arange(
            SEQUENCE_LENGTH,
            device=x.device
        )

        position_emb = self.position_embedding(
            positions
        )

        x = x + position_emb.unsqueeze(0)

        # ----------------------------------------------------
        # 5. Create causal attention mask
        # ----------------------------------------------------

        causal_mask = self.create_causal_mask(
            x.device
        )

        # ----------------------------------------------------
        # 6. Transformer
        #
        # padding_mask:
        # True  = padding
        # False = real timestep
        # ----------------------------------------------------

        hidden = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        # Expected:
        # (batch, 50, 128)

        # ----------------------------------------------------
        # 7. Create cognitive and affective representations
        # ----------------------------------------------------

        z_cog = self.cognitive_projection(
            hidden
        )

        z_aff = self.affective_projection(
            hidden
        )

        # ----------------------------------------------------
        # 8. Joint latent state
        # ----------------------------------------------------

        z = torch.cat(
            [
                z_cog,
                z_aff
            ],
            dim=-1
        )

        # Expected:
        # (batch, 50, 128)

        # ----------------------------------------------------
        # 9. Prediction heads
        # ----------------------------------------------------

        next_correctness = torch.sigmoid(
            self.correctness_head(z_cog).squeeze(-1)
        )

        next_affect = torch.sigmoid(
            self.affect_head(z_aff)
        )

        return {
            "hidden": hidden,
            "z_cog": z_cog,
            "z_aff": z_aff,
            "z": z,
            "next_correctness": next_correctness,
            "next_affect": next_affect
        }


# ============================================================
# Forward-pass test
# ============================================================

def main():

    print("Loading processed inputs...")

    model_inputs = np.load(
        "data/processed/model_inputs.npz"
    )

    categorical_inputs = np.load(
        "data/processed/categorical_inputs.npz"
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

    mask = model_inputs["mask"]

    print("X shape:", X.shape)
    print("Problem indices:", problem_indices.shape)
    print("Skill indices:", skill_indices.shape)
    print("Skill missing:", skill_missing.shape)
    print("Mask:", mask.shape)

    # --------------------------------------------------------
    # Select a small batch
    # --------------------------------------------------------

    X_batch = torch.tensor(
        X[:BATCH_SIZE],
        dtype=torch.float32
    )

    problem_batch = torch.tensor(
        problem_indices[:BATCH_SIZE],
        dtype=torch.long
    )

    skill_batch = torch.tensor(
        skill_indices[:BATCH_SIZE],
        dtype=torch.long
    )

    skill_missing_batch = torch.tensor(
        skill_missing[:BATCH_SIZE],
        dtype=torch.float32
    )

    mask_batch = torch.tensor(
        mask[:BATCH_SIZE],
        dtype=torch.bool
    )

    # Transformer expects:
    # True  = padding
    # False = real data

    padding_mask = ~mask_batch

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Vocabulary sizes
    # --------------------------------------------------------

    embedding_vocab = np.load(
        "data/processed/embedding_vocab.npz"
    )

    num_problems = len(
        embedding_vocab["problem_ids"]
    ) + 1

    num_skills = len(
        embedding_vocab["skill_ids"]
    ) + 1

    print("\nProblem vocabulary:", num_problems)
    print("Skill vocabulary:", num_skills)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = StudentEncoder(
        num_problems=num_problems,
        num_skills=num_skills
    ).to(device)

    model.eval()

    # --------------------------------------------------------
    # Move batch to GPU
    # --------------------------------------------------------

    X_batch = X_batch.to(device)
    problem_batch = problem_batch.to(device)
    skill_batch = skill_batch.to(device)
    skill_missing_batch = skill_missing_batch.to(device)
    padding_mask = padding_mask.to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print("\nRunning forward pass...")

    with torch.no_grad():

        outputs = model(
            numeric_features=X_batch,
            problem_indices=problem_batch,
            skill_indices=skill_batch,
            skill_missing=skill_missing_batch,
            padding_mask=padding_mask
        )

    # --------------------------------------------------------
    # Print output shapes
    # --------------------------------------------------------

    print("\n--- Output Shapes ---")

    for name, tensor in outputs.items():

        print(
            f"{name}: {tuple(tensor.shape)}"
        )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    assert outputs["hidden"].shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        D_MODEL
    )

    assert outputs["z_cog"].shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        COGNITIVE_DIM
    )

    assert outputs["z_aff"].shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        AFFECTIVE_DIM
    )

    assert outputs["z"].shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        COGNITIVE_DIM + AFFECTIVE_DIM
    )

    assert outputs["next_correctness"].shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH
    )

    assert outputs["next_affect"].shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        4
    )

    # Check numerical stability

    for name, tensor in outputs.items():

        assert torch.isfinite(
            tensor
        ).all(), f"{name} contains NaN or Inf"

    print("\nAll output shape checks passed.")
    print("All outputs are finite.")

    print("\nACDT PHASE 3.3 STUDENT ENCODER FORWARD PASS PASSED")


if __name__ == "__main__":
    main()