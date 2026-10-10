from pathlib import Path
import importlib

import numpy as np
import torch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_INPUTS_FILE = PROCESSED_DIR / "model_inputs.npz"
SEQUENCE_FILE = PROCESSED_DIR / "model_sequences.npz"
CATEGORICAL_FILE = PROCESSED_DIR / "categorical_inputs.npz"
VOCAB_FILE = PROCESSED_DIR / "embedding_vocab.npz"
SPLIT_FILE = PROCESSED_DIR / "student_split.npz"

CHECKPOINT_FILE = MODEL_DIR / "student_encoder_best.pt"

OUTPUT_FILE = PROCESSED_DIR / "world_model_transitions.npz"

BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ACDT PHASE 4.1 - WORLD MODEL TRANSITION PREPARATION")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed data...")

model_inputs = np.load(MODEL_INPUTS_FILE)
model_sequences = np.load(SEQUENCE_FILE)
categorical_inputs = np.load(CATEGORICAL_FILE)
vocab = np.load(VOCAB_FILE)
student_split = np.load(SPLIT_FILE)

X = model_inputs["X"]
mask = model_inputs["mask"]

targets = model_sequences["targets"]
target_mask = model_sequences["target_mask"]
user_ids = model_sequences["user_ids"]

problem_indices = categorical_inputs["problem_indices"]
skill_indices = categorical_inputs["skill_indices"]
skill_missing = categorical_inputs["skill_missing"]

train_users = student_split["train_users"]
val_users = student_split["val_users"]
test_users = student_split["test_users"]

print(f"X shape:              {X.shape}")
print(f"Mask shape:           {mask.shape}")
print(f"Target mask shape:    {target_mask.shape}")
print(f"Problem shape:        {problem_indices.shape}")
print(f"Skill shape:          {skill_indices.shape}")
print(f"Skill missing shape:  {skill_missing.shape}")
print(f"User IDs shape:       {user_ids.shape}")


# ============================================================
# BASIC CONSISTENCY CHECKS
# ============================================================

num_sequences, sequence_length, numeric_dim = X.shape

assert problem_indices.shape == (num_sequences, sequence_length)
assert skill_indices.shape == (num_sequences, sequence_length)
assert skill_missing.shape == (num_sequences, sequence_length)
assert target_mask.shape == (num_sequences, sequence_length)

print("\nBasic input consistency: PASSED")


# ============================================================
# LOAD STUDENT ENCODER
# ============================================================

print("\nLoading Student Encoder...")

student_encoder_module = importlib.import_module(
    "src.03_model.03_student_encoder"
)

StudentEncoder = student_encoder_module.StudentEncoder

num_problems = len(vocab["problem_ids"]) + 1
num_skills = len(vocab["skill_ids"]) + 1

model = StudentEncoder(
    num_problems=num_problems,
    num_skills=num_skills
).to(device)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

if not CHECKPOINT_FILE.exists():
    raise FileNotFoundError(
        f"Student Encoder checkpoint not found:\n{CHECKPOINT_FILE}"
    )

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_epoch = checkpoint.get("epoch", "unknown")
else:
    model.load_state_dict(checkpoint)
    checkpoint_epoch = "unknown"

model.eval()

# IMPORTANT:
# The Student Encoder is frozen during transition generation.
for parameter in model.parameters():
    parameter.requires_grad = False

print(f"Checkpoint epoch: {checkpoint_epoch}")
print(f"Problem vocabulary: {num_problems}")
print(f"Skill vocabulary:   {num_skills}")
print("Student Encoder frozen: YES")


# ============================================================
# DETERMINE STUDENT SPLIT FOR EACH SEQUENCE
# ============================================================

train_user_set = set(train_users.tolist())
val_user_set = set(val_users.tolist())
test_user_set = set(test_users.tolist())

sequence_split = np.empty(num_sequences, dtype=np.int8)

for i, user_id in enumerate(user_ids):
    if user_id in train_user_set:
        sequence_split[i] = 0
    elif user_id in val_user_set:
        sequence_split[i] = 1
    elif user_id in test_user_set:
        sequence_split[i] = 2
    else:
        raise ValueError(
            f"User {user_id} does not belong to any split."
        )

print("\nSequence split assignment: PASSED")

print(
    f"Train sequences: {np.sum(sequence_split == 0):,}"
)
print(
    f"Validation sequences: {np.sum(sequence_split == 1):,}"
)
print(
    f"Test sequences: {np.sum(sequence_split == 2):,}"
)


# ============================================================
# GENERATE LATENT STATES
# ============================================================

print("\nGenerating Student Encoder latent states...")

all_z = np.zeros(
    (num_sequences, sequence_length, 128),
    dtype=np.float32
)

num_batches = (num_sequences + BATCH_SIZE - 1) // BATCH_SIZE

with torch.no_grad():

    for batch_number, start in enumerate(
        range(0, num_sequences, BATCH_SIZE),
        start=1
    ):

        end = min(start + BATCH_SIZE, num_sequences)

        numeric_batch = torch.from_numpy(
            X[start:end]
        ).float().to(device)

        problem_batch = torch.from_numpy(
            problem_indices[start:end]
        ).long().to(device)

        skill_batch = torch.from_numpy(
            skill_indices[start:end]
        ).long().to(device)

        skill_missing_batch = torch.from_numpy(
            skill_missing[start:end]
        ).float().to(device)

        padding_mask_batch = torch.from_numpy(
            mask[start:end] == 0
        ).to(device)

        outputs = model(
            numeric_features=numeric_batch,
            problem_indices=problem_batch,
            skill_indices=skill_batch,
            skill_missing=skill_missing_batch,
            padding_mask=padding_mask_batch
        )

        z_batch = outputs["z"]

        all_z[start:end] = (
            z_batch.cpu().numpy().astype(np.float32)
        )

        if batch_number == 1 or batch_number % 500 == 0:
            print(
                f"Processed batches: "
                f"{batch_number:,}/{num_batches:,}"
            )


print(f"Latent state shape: {all_z.shape}")

if not np.isfinite(all_z).all():
    raise ValueError("Latent states contain NaN or infinite values.")

print("Latent states finite: PASSED")


# ============================================================
# CONSTRUCT TRANSITIONS
# ============================================================

print("\nConstructing valid state transitions...")

# target_mask[t] == 1 means that interaction t has a valid
# next interaction.
#
# Therefore:
#
# current state = z[t]
# action         = problem/skill at t+1
# next state     = z[t+1]
#
# This matches the later counterfactual-planning interpretation:
# "What happens if we choose this next problem?"

valid_transition_mask = target_mask.astype(bool)

transition_sequence_indices, transition_time_indices = np.where(
    valid_transition_mask
)

num_transitions = len(transition_sequence_indices)

print(f"Valid transitions: {num_transitions:,}")


# ============================================================
# EXTRACT TRANSITION ARRAYS
# ============================================================

sequence_indices = transition_sequence_indices
time_indices = transition_time_indices

current_states = all_z[
    sequence_indices,
    time_indices
]

next_states = all_z[
    sequence_indices,
    time_indices + 1
]

# IMPORTANT:
# The action is the NEXT interaction.
action_problem_indices = problem_indices[
    sequence_indices,
    time_indices + 1
]

action_skill_indices = skill_indices[
    sequence_indices,
    time_indices + 1
]

action_skill_missing = skill_missing[
    sequence_indices,
    time_indices + 1
]

transition_user_ids = user_ids[
    sequence_indices
]

transition_splits = sequence_split[
    sequence_indices
]


# ============================================================
# VERIFY TRANSITIONS
# ============================================================

print("\nVerifying transition alignment...")

if current_states.shape != (
    num_transitions,
    128
):
    raise ValueError(
        f"Unexpected current state shape: "
        f"{current_states.shape}"
    )

if next_states.shape != (
    num_transitions,
    128
):
    raise ValueError(
        f"Unexpected next state shape: "
        f"{next_states.shape}"
    )

if action_problem_indices.shape != (num_transitions,):
    raise ValueError("Unexpected problem action shape.")

if action_skill_indices.shape != (num_transitions,):
    raise ValueError("Unexpected skill action shape.")

if action_skill_missing.shape != (num_transitions,):
    raise ValueError("Unexpected skill-missing action shape.")

if not np.isfinite(current_states).all():
    raise ValueError("Current states contain invalid values.")

if not np.isfinite(next_states).all():
    raise ValueError("Next states contain invalid values.")

print("Transition shapes: PASSED")
print("Transition values finite: PASSED")


# ============================================================
# CHECK SPLIT COUNTS
# ============================================================

train_count = np.sum(transition_splits == 0)
val_count = np.sum(transition_splits == 1)
test_count = np.sum(transition_splits == 2)

print("\nTransition split:")
print(f"Train transitions:      {train_count:,}")
print(f"Validation transitions: {val_count:,}")
print(f"Test transitions:       {test_count:,}")

if train_count == 0:
    raise ValueError("No training transitions found.")

if val_count == 0:
    raise ValueError("No validation transitions found.")

if test_count == 0:
    raise ValueError("No test transitions found.")


# ============================================================
# CHECK USER SPLIT ISOLATION
# ============================================================

train_transition_users = set(
    transition_user_ids[transition_splits == 0].tolist()
)

val_transition_users = set(
    transition_user_ids[transition_splits == 1].tolist()
)

test_transition_users = set(
    transition_user_ids[transition_splits == 2].tolist()
)

if train_transition_users & val_transition_users:
    raise ValueError("Train/validation user overlap detected.")

if train_transition_users & test_transition_users:
    raise ValueError("Train/test user overlap detected.")

if val_transition_users & test_transition_users:
    raise ValueError("Validation/test user overlap detected.")

print("Student split isolation: PASSED")


# ============================================================
# CHECK SKILL MISSINGNESS
# ============================================================

print("\nAction skill statistics:")

missing_count = np.sum(action_skill_missing == 1)
known_count = np.sum(action_skill_missing == 0)

print(f"Known skill actions:    {known_count:,}")
print(f"Missing skill actions:  {missing_count:,}")

if np.any(action_skill_indices < 0):
    raise ValueError("Negative skill indices found.")

if np.any(action_problem_indices < 0):
    raise ValueError("Negative problem indices found.")

print("Action indices valid: PASSED")


# ============================================================
# SAVE
# ============================================================

print("\nSaving World Model transition dataset...")

np.savez_compressed(
    OUTPUT_FILE,

    current_states=current_states.astype(np.float32),

    next_states=next_states.astype(np.float32),

    action_problem_indices=(
        action_problem_indices.astype(np.int32)
    ),

    action_skill_indices=(
        action_skill_indices.astype(np.int32)
    ),

    action_skill_missing=(
        action_skill_missing.astype(np.int8)
    ),

    user_ids=transition_user_ids,

    split=transition_splits.astype(np.int8),

    sequence_indices=sequence_indices.astype(np.int32),

    time_indices=time_indices.astype(np.int16)
)

print(f"Saved: {OUTPUT_FILE}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ACDT PHASE 4.1 TRANSITION PREPARATION PASSED")
print("=" * 60)

print(f"Total transitions: {num_transitions:,}")
print(f"Current states:    {current_states.shape}")
print(f"Next states:       {next_states.shape}")
print(f"Problem actions:   {action_problem_indices.shape}")
print(f"Skill actions:     {action_skill_indices.shape}")

print("\nSplit:")
print(f"  Train:      {train_count:,}")
print(f"  Validation: {val_count:,}")
print(f"  Test:       {test_count:,}")

print("\nStudent Encoder was used only to generate frozen states.")
print("No World Model training has been performed yet.")