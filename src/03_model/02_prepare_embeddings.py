import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_FILE = "data/processed/model_sequences.npz"
OUTPUT_FILE = "data/processed/categorical_inputs.npz"
VOCAB_FILE = "data/processed/embedding_vocab.npz"


# ---------------------------------------------------------
# Load model sequences
# ---------------------------------------------------------

print("Loading model sequences...")

data = np.load(INPUT_FILE, allow_pickle=True)

problem_ids = data["problem_ids"]
skill_ids = data["skill_ids"]
mask = data["mask"]
user_ids = data["user_ids"]

print(f"Problem ID shape: {problem_ids.shape}")
print(f"Skill ID shape:   {skill_ids.shape}")
print(f"Mask shape:       {mask.shape}")
print(f"User ID shape:    {user_ids.shape}")


# ---------------------------------------------------------
# Basic validation
# ---------------------------------------------------------

assert problem_ids.shape == mask.shape
assert skill_ids.shape == mask.shape
assert user_ids.shape[0] == problem_ids.shape[0]

print("\nBasic shape validation passed.")


# ---------------------------------------------------------
# Identify real positions
# ---------------------------------------------------------

real_positions = mask == 1

real_problem_ids = problem_ids[real_positions]
real_skill_ids = skill_ids[real_positions]

print(f"\nReal interactions: {len(real_problem_ids):,}")


# ---------------------------------------------------------
# Problem vocabulary
# ---------------------------------------------------------

print("\n--- Building problem vocabulary ---")

unique_problem_ids = np.unique(real_problem_ids)

print(f"Unique problems: {len(unique_problem_ids):,}")

# Reserve index 0 for unknown/padding
problem_to_index = {
    problem_id: index
    for index, problem_id in enumerate(unique_problem_ids, start=1)
}

print(f"Problem vocabulary size including reserved index: "
      f"{len(problem_to_index) + 1:,}")


# ---------------------------------------------------------
# Skill vocabulary
# ---------------------------------------------------------

print("\n--- Building skill vocabulary ---")

# skill_id == -1 represents missing skill in model_sequences.npz
known_skill_ids = real_skill_ids[real_skill_ids != -1]

unique_skill_ids = np.unique(known_skill_ids)

print(f"Unique known skills: {len(unique_skill_ids):,}")

# Reserve:
# 0 = UNKNOWN / missing skill
# 1+ = actual skill IDs

skill_to_index = {
    skill_id: index
    for index, skill_id in enumerate(unique_skill_ids, start=1)
}

print("Skill index 0 reserved for UNKNOWN / missing skill.")
print(f"Skill vocabulary size: {len(skill_to_index) + 1:,}")


# ---------------------------------------------------------
# Convert problem IDs to indices
# ---------------------------------------------------------

print("\n--- Converting problem IDs to indices ---")

# unique_problem_ids is already sorted and contains the
# actual problem IDs.
#
# searchsorted finds the position of each problem ID
# inside that sorted vocabulary.
problem_indices = (
    np.searchsorted(unique_problem_ids, problem_ids) + 1
).astype(np.int32)

# Padding positions must remain 0.
problem_indices[mask == 0] = 0

print("Problem ID conversion complete.")


# ---------------------------------------------------------
# Convert skill IDs to indices
# ---------------------------------------------------------

print("\n--- Converting skill IDs to indices ---")

skill_indices = np.zeros(skill_ids.shape, dtype=np.int32)

for skill_id, index in skill_to_index.items():
    skill_indices[skill_ids == skill_id] = index


# ---------------------------------------------------------
# Explicit missing-skill indicator
# ---------------------------------------------------------

skill_missing = (
    (skill_ids == -1) & (mask == 1)
).astype(np.float32)

print(
    f"Missing-skill interactions: "
    f"{int(skill_missing.sum()):,}"
)


# ---------------------------------------------------------
# Padding validation
# ---------------------------------------------------------

print("\n--- Padding validation ---")

padding_positions = mask == 0

print(
    "Maximum problem index at padding:",
    problem_indices[padding_positions].max()
)

print(
    "Maximum skill index at padding:",
    skill_indices[padding_positions].max()
)

print(
    "Maximum skill-missing flag at padding:",
    skill_missing[padding_positions].max()
)


# ---------------------------------------------------------
# Save categorical inputs
# ---------------------------------------------------------

np.savez_compressed(
    OUTPUT_FILE,
    problem_indices=problem_indices,
    skill_indices=skill_indices,
    skill_missing=skill_missing
)


# ---------------------------------------------------------
# Save vocabularies
# ---------------------------------------------------------

problem_ids_sorted = np.array(
    list(problem_to_index.keys())
)

problem_indices_sorted = np.array(
    list(problem_to_index.values())
)

skill_ids_sorted = np.array(
    list(skill_to_index.keys())
)

skill_indices_sorted = np.array(
    list(skill_to_index.values())
)

np.savez_compressed(
    VOCAB_FILE,
    problem_ids=problem_ids_sorted,
    problem_indices=problem_indices_sorted,
    skill_ids=skill_ids_sorted,
    skill_indices=skill_indices_sorted
)


# ---------------------------------------------------------
# Final validation
# ---------------------------------------------------------

print("\n--- Final validation ---")

print(f"Problem indices shape: {problem_indices.shape}")
print(f"Skill indices shape:   {skill_indices.shape}")
print(f"Skill missing shape:   {skill_missing.shape}")

print(
    "Problem index range:",
    problem_indices.min(),
    "to",
    problem_indices.max()
)

print(
    "Skill index range:",
    skill_indices.min(),
    "to",
    skill_indices.max()
)

print(
    "Problem indices at padding:",
    np.unique(problem_indices[padding_positions])
)

print(
    "Skill indices at padding:",
    np.unique(skill_indices[padding_positions])
)

print(
    "Skill missing at padding:",
    np.unique(skill_missing[padding_positions])
)

assert np.all(problem_indices[padding_positions] == 0)
assert np.all(skill_indices[padding_positions] == 0)
assert np.all(skill_missing[padding_positions] == 0)

assert np.all(problem_indices[real_positions] > 0)
assert np.all(skill_indices[real_positions][real_skill_ids != -1] > 0)
assert np.all(skill_indices[real_positions][real_skill_ids == -1] == 0)

assert int(skill_missing.sum()) == int((real_skill_ids == -1).sum())

print("\nACDT PHASE 3.2 INPUT PREPARATION PASSED")