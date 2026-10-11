
import numpy as np
import torch
from pathlib import Path

from importlib.util import spec_from_file_location, module_from_spec


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(".")
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "results" / "mcts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "test_initial_state.npz"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# LOAD STUDENT ENCODER CLASS
# ============================================================

module_path = ROOT / "src" / "03_model" / "03_student_encoder.py"

spec = spec_from_file_location("student_encoder_module", module_path)
encoder_module = module_from_spec(spec)
spec.loader.exec_module(encoder_module)

StudentEncoder = encoder_module.StudentEncoder


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading model inputs...")

model_data = np.load(DATA_DIR / "model_inputs.npz")
categorical_data = np.load(DATA_DIR / "categorical_inputs.npz")
vocab = np.load(DATA_DIR / "embedding_vocab.npz", allow_pickle=True)
split_data = np.load(DATA_DIR / "student_split.npz")

X = model_data["X"]
mask = model_data["mask"]
user_ids = model_data["user_ids"]

problem_indices = categorical_data["problem_indices"]
skill_indices = categorical_data["skill_indices"]
skill_missing = categorical_data["skill_missing"]

train_users = split_data["train_users"]
val_users = split_data["val_users"]
test_users = split_data["test_users"]

print("Sequence count:", len(X))
print("Numeric input shape:", X.shape)
print("Held-out test users:", len(test_users))


# ============================================================
# SELECT A REAL TEST SEQUENCE
# ============================================================

test_user_set = set(test_users.tolist())

candidate_sequences = np.flatnonzero(
    np.isin(user_ids, list(test_user_set))
    & (mask.sum(axis=1) > 0)
)

if len(candidate_sequences) == 0:
    raise RuntimeError("No valid held-out test sequences found.")

# Select the first eligible sequence deterministically.
sequence_index = int(candidate_sequences[0])
selected_user = int(user_ids[sequence_index])

if selected_user in set(train_users.tolist()):
    raise AssertionError("Selected user belongs to training split.")

if selected_user in set(val_users.tolist()):
    raise AssertionError("Selected user belongs to validation split.")

print("\nSelected test sequence:", sequence_index)
print("Selected test user:", selected_user)

valid_length = int(mask[sequence_index].sum())

if not 1 <= valid_length <= X.shape[1]:
    raise ValueError(f"Unexpected valid sequence length: {valid_length}")

last_timestep = valid_length - 1

print("Valid interactions in window:", valid_length)
print("Last valid timestep:", last_timestep)


# ============================================================
# INITIALIZE AND LOAD TRAINED ENCODER
# ============================================================

num_problems = int(vocab["problem_indices"].max()) + 1
num_skills = int(vocab["skill_indices"].max()) + 1

print("\nProblem embedding vocabulary size:", num_problems)
print("Skill embedding vocabulary size:", num_skills)

model = StudentEncoder(
    num_problems=num_problems,
    num_skills=num_skills
).to(DEVICE)

checkpoint_path = MODEL_DIR / "student_encoder_best.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

# Support either a raw state_dict or a checkpoint dictionary.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print("Loaded trained Student Encoder.")


# ============================================================
# ENCODE SELECTED STUDENT HISTORY
# ============================================================

numeric_tensor = torch.from_numpy(
    X[sequence_index:sequence_index + 1].astype(np.float32)
).to(DEVICE)

problem_tensor = torch.from_numpy(
    problem_indices[sequence_index:sequence_index + 1].astype(np.int64)
).to(DEVICE)

skill_tensor = torch.from_numpy(
    skill_indices[sequence_index:sequence_index + 1].astype(np.int64)
).to(DEVICE)

missing_tensor = torch.from_numpy(
    skill_missing[sequence_index:sequence_index + 1].astype(np.float32)
).to(DEVICE)

# Encoder convention: True means padding.
padding_mask = torch.from_numpy(
    (mask[sequence_index:sequence_index + 1] == 0)
).to(DEVICE)

with torch.no_grad():
    outputs = model(
        numeric_features=numeric_tensor,
        problem_indices=problem_tensor,
        skill_indices=skill_tensor,
        skill_missing=missing_tensor,
        padding_mask=padding_mask
    )

    # Use the last real interaction, not a padded timestep.
    z = outputs["z"][0, last_timestep]
    z_cog = outputs["z_cog"][0, last_timestep]
    z_aff = outputs["z_aff"][0, last_timestep]

    predicted_correctness = outputs[
        "next_correctness"
    ][0, last_timestep]

    predicted_affect = outputs[
        "next_affect"
    ][0, last_timestep]

z = z.detach().cpu().numpy().astype(np.float32)
z_cog = z_cog.detach().cpu().numpy().astype(np.float32)
z_aff = z_aff.detach().cpu().numpy().astype(np.float32)

predicted_correctness = float(predicted_correctness.item())
predicted_affect = predicted_affect.detach().cpu().numpy().astype(np.float32)


# ============================================================
# VALIDATE AND SAVE
# ============================================================

assert z.shape == (128,)
assert z_cog.shape == (64,)
assert z_aff.shape == (64,)
assert predicted_affect.shape == (4,)
assert np.isfinite(z).all()
assert np.isfinite(z_cog).all()
assert np.isfinite(z_aff).all()
assert np.isfinite(predicted_correctness)
assert np.isfinite(predicted_affect).all()

np.savez_compressed(
    OUTPUT_FILE,
    initial_state=z,
    cognitive_state=z_cog,
    affective_state=z_aff,
    predicted_next_correctness=np.array(predicted_correctness),
    predicted_next_affect=predicted_affect,
    user_id=np.array(selected_user),
    sequence_index=np.array(sequence_index),
    valid_length=np.array(valid_length),
    last_timestep=np.array(last_timestep)
)

print("\n========== INITIAL STATE VALIDATION ==========")
print("Latent state shape:", z.shape)
print("Cognitive state shape:", z_cog.shape)
print("Affective state shape:", z_aff.shape)
print(f"Predicted next correctness: {predicted_correctness:.4f}")

affect_names = [
    "Frustrated",
    "Confused",
    "Concentrating",
    "Bored"
]

print("Predicted next affect:")
for name, value in zip(affect_names, predicted_affect):
    print(f"  {name}: {value:.4f}")

print("All latent values finite:", bool(np.isfinite(z).all()))
print("Saved to:", OUTPUT_FILE)
print("\nInitial student state prepared successfully.")
