
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(".")
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MCTS_DIR = ROOT / "results" / "mcts"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

AFFECT_NAMES = [
    "frustration",
    "confusion",
    "concentration",
    "boredom",
]

# Initial, provisional reward weights.
W_CONCENTRATION = 0.15
W_FRUSTRATION = 0.05
W_CONFUSION = 0.05
W_BOREDOM = 0.025
W_UNCERTAINTY = 0.05


# ============================================================
# LOAD MODEL CLASSES
# ============================================================

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


encoder_module = load_module(
    "student_encoder_module",
    ROOT / "src" / "03_model" / "03_student_encoder.py",
)

world_module = load_module(
    "world_model_module",
    ROOT / "src" / "03_model" / "08_world_model.py",
)

StudentEncoder = encoder_module.StudentEncoder
WorldModel = world_module.WorldModel


def load_checkpoint(path, model):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()


# ============================================================
# LOAD VOCABULARY AND MODELS
# ============================================================

print("Device:", DEVICE)
print("Loading vocabulary...")

vocab = np.load(
    DATA_DIR / "embedding_vocab.npz",
    allow_pickle=True,
)

num_problems = int(vocab["problem_indices"].max()) + 1
num_skills = int(vocab["skill_indices"].max()) + 1

encoder = StudentEncoder(
    num_problems=num_problems,
    num_skills=num_skills,
).to(DEVICE)

world_model = WorldModel(
    num_problems=num_problems,
    num_skills=num_skills,
).to(DEVICE)

load_checkpoint(
    MODEL_DIR / "student_encoder_best.pt",
    encoder,
)

load_checkpoint(
    MODEL_DIR / "world_model_best.pt",
    world_model,
)

print("Both trained models loaded.")


# ============================================================
# LOAD INITIAL STATE AND CANDIDATE ACTIONS
# ============================================================

initial = np.load(
    MCTS_DIR / "test_initial_state.npz",
)

catalogue = np.load(
    MCTS_DIR / "candidate_catalogue.npz",
    allow_pickle=True,
)

current_state = torch.tensor(
    initial["initial_state"],
    dtype=torch.float32,
    device=DEVICE,
).unsqueeze(0)

current_correctness = float(
    initial["predicted_next_correctness"]
)

current_affect = np.asarray(
    initial["predicted_next_affect"],
    dtype=np.float32,
)

problem_indices = torch.tensor(
    catalogue["problem_indices"],
    dtype=torch.long,
    device=DEVICE,
)

skill_indices = torch.tensor(
    catalogue["skill_indices"],
    dtype=torch.long,
    device=DEVICE,
)

skill_missing = torch.tensor(
    catalogue["skill_missing"],
    dtype=torch.float32,
    device=DEVICE,
)

candidate_count = len(problem_indices)

if candidate_count == 0:
    raise ValueError("Candidate catalogue is empty.")

print("Candidate problems:", candidate_count)


# ============================================================
# SIMULATE ONE STEP FOR EVERY CANDIDATE
# ============================================================

print("Predicting candidate next states...")

with torch.no_grad():
    states = current_state.expand(candidate_count, -1)

    predictions = world_model(
        current_state=states,
        problem_indices=problem_indices,
        skill_indices=skill_indices,
        skill_missing=skill_missing,
    )

    next_states = predictions["next_state_mean"]
    next_state_std = predictions["next_state_std"]

    # Joint state = 64D cognitive + 64D affective.
    next_cognitive = next_states[:, :64]
    next_affective = next_states[:, 64:]

    # Apply the trained encoder's prediction heads directly
    # to the corresponding halves of the simulated state.
    predicted_correctness = torch.sigmoid(
        encoder.correctness_head(next_cognitive).squeeze(-1)
    )

    predicted_affect = torch.sigmoid(
        encoder.affect_head(next_affective)
    )

    mean_state_std = next_state_std.mean(dim=1)

correctness = predicted_correctness.cpu().numpy()
affect = predicted_affect.cpu().numpy()
uncertainty = mean_state_std.cpu().numpy()


# ============================================================
# CALCULATE TRANSPARENT REWARD COMPONENTS
# ============================================================

delta_correctness = correctness - current_correctness

delta_frustration = affect[:, 0] - current_affect[0]
delta_confusion = affect[:, 1] - current_affect[1]
delta_concentration = affect[:, 2] - current_affect[2]
delta_boredom = affect[:, 3] - current_affect[3]

reward_correctness = delta_correctness
reward_concentration = W_CONCENTRATION * delta_concentration
reward_frustration = -W_FRUSTRATION * delta_frustration
reward_confusion = -W_CONFUSION * delta_confusion
reward_boredom = -W_BOREDOM * delta_boredom
reward_uncertainty = -W_UNCERTAINTY * uncertainty

total_reward = (
    reward_correctness
    + reward_concentration
    + reward_frustration
    + reward_confusion
    + reward_boredom
    + reward_uncertainty
)


# ============================================================
# RANK CANDIDATES AND SAVE RESULTS
# ============================================================

original_problem_ids = catalogue["original_problem_ids"].astype(str)
original_skill_ids = catalogue["original_skill_ids"].astype(str)
frequency = catalogue["interaction_frequency"]

results = pd.DataFrame({
    "problem_index": catalogue["problem_indices"],
    "original_problem_id": original_problem_ids,
    "skill_index": catalogue["skill_indices"],
    "original_skill_id": original_skill_ids,
    "interaction_frequency": frequency,
    "predicted_correctness": correctness,
    "predicted_frustration": affect[:, 0],
    "predicted_confusion": affect[:, 1],
    "predicted_concentration": affect[:, 2],
    "predicted_boredom": affect[:, 3],
    "mean_latent_std": uncertainty,
    "reward_correctness": reward_correctness,
    "reward_concentration": reward_concentration,
    "reward_frustration": reward_frustration,
    "reward_confusion": reward_confusion,
    "reward_boredom": reward_boredom,
    "reward_uncertainty": reward_uncertainty,
    "total_reward": total_reward,
})

results = results.sort_values(
    "total_reward",
    ascending=False,
).reset_index(drop=True)

results.insert(0, "rank", np.arange(1, len(results) + 1))

output_path = MCTS_DIR / "reward_test_results.csv"
results.to_csv(output_path, index=False)


# ============================================================
# VALIDATE AND REPORT
# ============================================================

numeric_results = results.select_dtypes(include=[np.number]).to_numpy()

assert len(results) == candidate_count
assert np.isfinite(numeric_results).all()
assert np.isfinite(total_reward).all()
assert np.all(uncertainty > 0)

print("\n========== REWARD TEST SUMMARY ==========")
print("Candidates scored:", len(results))
print(f"Current predicted correctness: {current_correctness:.4f}")

print("\nCurrent predicted affect:")
for name, value in zip(AFFECT_NAMES, current_affect):
    print(f"  {name}: {value:.4f}")

print("\nReward distribution:")
print(f"  Minimum: {total_reward.min():.6f}")
print(f"  Maximum: {total_reward.max():.6f}")
print(f"  Mean:    {total_reward.mean():.6f}")
print(f"  Std:     {total_reward.std():.6f}")

print("\nTop 10 candidates:")
columns = [
    "rank",
    "original_problem_id",
    "original_skill_id",
    "predicted_correctness",
    "predicted_concentration",
    "predicted_frustration",
    "mean_latent_std",
    "total_reward",
]

print(results[columns].head(10).to_string(index=False))

print("\nSaved results:", output_path)
print("\nReward-function test completed successfully.")
print(
    "Note: rankings are exploratory model estimates, "
    "not validated causal learning recommendations."
)
