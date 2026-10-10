import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    Affective-Cognitive Digital Twin World Model.

    Given:
        - current latent student state z_t
        - candidate problem
        - candidate skill
        - skill-missing indicator

    predicts:
        - mean of next latent state
        - uncertainty of next latent state
    """

    # ========================================================
    # MODEL DIMENSIONS
    # ========================================================

    STATE_DIM = 128

    PROBLEM_EMBED_DIM = 32
    SKILL_EMBED_DIM = 16

    HIDDEN_DIM = 256

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, num_problems, num_skills):

        super().__init__()

        self.num_problems = num_problems
        self.num_skills = num_skills

        # ----------------------------------------------------
        # Action embeddings
        # ----------------------------------------------------

        self.problem_embedding = nn.Embedding(
            num_problems,
            self.PROBLEM_EMBED_DIM,
            padding_idx=0
        )

        self.skill_embedding = nn.Embedding(
            num_skills,
            self.SKILL_EMBED_DIM,
            padding_idx=0
        )

        # ----------------------------------------------------
        # State + action representation
        #
        # 128 state
        # + 32 problem
        # + 16 skill
        # + 1 missing indicator
        # = 177 dimensions
        # ----------------------------------------------------

        input_dim = (
            self.STATE_DIM
            + self.PROBLEM_EMBED_DIM
            + self.SKILL_EMBED_DIM
            + 1
        )

        # ----------------------------------------------------
        # Transition network
        # ----------------------------------------------------

        self.transition_network = nn.Sequential(
            nn.Linear(input_dim, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # ----------------------------------------------------
        # Mean prediction
        #
        # Predicts expected next latent state μ_(t+1)
        # ----------------------------------------------------

        self.mean_head = nn.Linear(
            self.HIDDEN_DIM,
            self.STATE_DIM
        )

        # ----------------------------------------------------
        # Uncertainty prediction
        #
        # Produces a raw value that is converted to positive
        # standard deviation using Softplus.
        # ----------------------------------------------------

        self.uncertainty_head = nn.Linear(
            self.HIDDEN_DIM,
            self.STATE_DIM
        )

        self.softplus = nn.Softplus()

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(
        self,
        current_state,
        problem_indices,
        skill_indices,
        skill_missing
    ):
        """
        Parameters
        ----------
        current_state:
            Tensor of shape (B, 128)

        problem_indices:
            Tensor of shape (B,)

        skill_indices:
            Tensor of shape (B,)

        skill_missing:
            Tensor of shape (B,)

        Returns
        -------
        Dictionary containing:

        next_state_mean:
            Predicted μ_(t+1), shape (B, 128)

        next_state_std:
            Predicted σ_(t+1), shape (B, 128)
        """

        # ----------------------------------------------------
        # Embed candidate action
        # ----------------------------------------------------

        problem_embedding = self.problem_embedding(
            problem_indices
        )

        skill_embedding = self.skill_embedding(
            skill_indices
        )

        # Ensure missing indicator has shape (B, 1)
        skill_missing = skill_missing.float().unsqueeze(-1)

        # ----------------------------------------------------
        # Combine current state + candidate action
        # ----------------------------------------------------

        transition_input = torch.cat(
            [
                current_state,
                problem_embedding,
                skill_embedding,
                skill_missing
            ],
            dim=-1
        )

        # ----------------------------------------------------
        # Transition network
        # ----------------------------------------------------

        hidden = self.transition_network(
            transition_input
        )

        # ----------------------------------------------------
        # Predict next-state mean
        # ----------------------------------------------------

        next_state_mean = self.mean_head(hidden)

        # ----------------------------------------------------
        # Predict uncertainty
        #
        # Softplus guarantees:
        #
        # sigma > 0
        # ----------------------------------------------------

        next_state_std = (
            self.softplus(
                self.uncertainty_head(hidden)
            )
            + 1e-6
        )

        return {
            "next_state_mean": next_state_mean,
            "next_state_std": next_state_std
        }


# ============================================================
# FORWARD-PASS TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ACDT PHASE 4.2 - WORLD MODEL FORWARD TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # These are the actual vocab sizes from Phase 3.2.
    # --------------------------------------------------------

    num_problems = 180000
    num_skills = 266

    model = WorldModel(
        num_problems=num_problems,
        num_skills=num_skills
    ).to(device)

    # --------------------------------------------------------
    # Dummy batch matching the real World Model inputs.
    # --------------------------------------------------------

    batch_size = 32

    current_state = torch.randn(
        batch_size,
        128,
        device=device
    )

    problem_indices = torch.randint(
        low=1,
        high=num_problems,
        size=(batch_size,),
        device=device
    )

    skill_indices = torch.randint(
        low=0,
        high=num_skills,
        size=(batch_size,),
        device=device
    )

    skill_missing = torch.randint(
        low=0,
        high=2,
        size=(batch_size,),
        device=device
    ).float()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        outputs = model(
            current_state=current_state,
            problem_indices=problem_indices,
            skill_indices=skill_indices,
            skill_missing=skill_missing
        )

    next_state_mean = outputs[
        "next_state_mean"
    ]

    next_state_std = outputs[
        "next_state_std"
    ]

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print("\nInput shapes:")
    print(
        f"Current state:    {current_state.shape}"
    )
    print(
        f"Problem indices:  {problem_indices.shape}"
    )
    print(
        f"Skill indices:    {skill_indices.shape}"
    )
    print(
        f"Skill missing:    {skill_missing.shape}"
    )

    print("\nOutput shapes:")
    print(
        f"Next-state mean:  {next_state_mean.shape}"
    )
    print(
        f"Next-state std:   {next_state_std.shape}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert next_state_mean.shape == (
        batch_size,
        128
    )

    assert next_state_std.shape == (
        batch_size,
        128
    )

    assert torch.isfinite(
        next_state_mean
    ).all()

    assert torch.isfinite(
        next_state_std
    ).all()

    assert torch.all(
        next_state_std > 0
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel:")
    print(
        f"Total parameters:     {total_parameters:,}"
    )
    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    print("\nUncertainty:")
    print(
        f"Minimum predicted std: "
        f"{next_state_std.min().item():.6f}"
    )
    print(
        f"Maximum predicted std: "
        f"{next_state_std.max().item():.6f}"
    )

    print("\n" + "=" * 60)
    print("ACDT PHASE 4.2 FORWARD PASS PASSED")
    print("=" * 60)