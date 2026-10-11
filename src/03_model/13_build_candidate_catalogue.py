
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/mcts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 1000

CATEGORICAL_FILE = BASE_DIR / "categorical_inputs.npz"
VOCAB_FILE = BASE_DIR / "embedding_vocab.npz"

OUTPUT_NPZ = OUTPUT_DIR / "candidate_catalogue.npz"
OUTPUT_CSV = OUTPUT_DIR / "candidate_catalogue.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading categorical inputs...")

data = np.load(CATEGORICAL_FILE)

problems = data["problem_indices"].reshape(-1)
skills = data["skill_indices"].reshape(-1)
missing = data["skill_missing"].reshape(-1)

print("Loading embedding vocabulary...")

vocab = np.load(VOCAB_FILE, allow_pickle=True)

problem_ids = vocab["problem_ids"]
problem_indices = vocab["problem_indices"]

skill_ids = vocab["skill_ids"]
skill_indices = vocab["skill_indices"]

# Model index -> original dataset ID
problem_id_map = {
    int(index): str(original_id)
    for original_id, index in zip(problem_ids, problem_indices)
}

skill_id_map = {
    int(index): str(original_id)
    for original_id, index in zip(skill_ids, skill_indices)
}


# ============================================================
# COUNT OBSERVED PROBLEMS
# ============================================================

print("Counting problem frequencies...")

valid_problem = problems > 0

total_frequency = np.bincount(
    problems[valid_problem],
    minlength=int(problems.max()) + 1
)

known_skill = (
    valid_problem
    & (skills > 0)
    & (missing == 0)
)

# Count problem-skill pairs observed with known skills.
pairs = np.unique(
    np.column_stack((problems[known_skill], skills[known_skill])),
    axis=0
)

# Every known-skill problem should have exactly one known skill,
# based on the preceding catalogue analysis.
unique_problem_ids, pair_counts = np.unique(
    pairs[:, 0],
    return_counts=True
)

conflicting = unique_problem_ids[pair_counts > 1]

if len(conflicting):
    raise ValueError(
        f"Found {len(conflicting)} problems with multiple known skills. "
        "Inspect the associations before building the catalogue."
    )

# Sort by total observed frequency, highest first.
ranked_problems = unique_problem_ids[
    np.argsort(total_frequency[unique_problem_ids])[::-1]
]

selected = ranked_problems[:TOP_K]

if len(selected) < TOP_K:
    print(
        f"Note: only {len(selected)} eligible problems were found "
        f"for the requested TOP_K={TOP_K}."
    )


# ============================================================
# BUILD CATALOGUE
# ============================================================

print("Building candidate catalogue...")

# Map each known-skill problem to its observed skill index.
problem_to_skill = {
    int(problem): int(skill)
    for problem, skill in pairs
}

catalogue_problem_indices = selected.astype(np.int32)

catalogue_skill_indices = np.array(
    [problem_to_skill[int(p)] for p in selected],
    dtype=np.int32
)

catalogue_skill_missing = np.zeros(
    len(selected),
    dtype=np.float32
)

catalogue_frequency = total_frequency[selected].astype(np.int64)

catalogue_problem_ids = np.array(
    [problem_id_map[int(p)] for p in selected],
    dtype=str
)

catalogue_skill_ids = np.array(
    [skill_id_map[int(s)] for s in catalogue_skill_indices],
    dtype=str
)

# Save the arrays required by the future planner.
np.savez_compressed(
    OUTPUT_NPZ,
    problem_indices=catalogue_problem_indices,
    skill_indices=catalogue_skill_indices,
    skill_missing=catalogue_skill_missing,
    interaction_frequency=catalogue_frequency,
    original_problem_ids=catalogue_problem_ids,
    original_skill_ids=catalogue_skill_ids
)

# Save a human-readable copy for inspection.
catalogue_df = pd.DataFrame({
    "problem_index": catalogue_problem_indices,
    "original_problem_id": catalogue_problem_ids,
    "skill_index": catalogue_skill_indices,
    "original_skill_id": catalogue_skill_ids,
    "skill_missing": catalogue_skill_missing.astype(int),
    "interaction_frequency": catalogue_frequency
})

catalogue_df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# VALIDATION
# ============================================================

saved = np.load(OUTPUT_NPZ)

assert len(saved["problem_indices"]) == len(selected)
assert np.all(saved["problem_indices"] > 0)
assert np.all(saved["skill_indices"] > 0)
assert np.all(saved["skill_missing"] == 0)
assert len(np.unique(saved["problem_indices"])) == len(selected)
assert np.all(saved["interaction_frequency"] > 0)

print("\n========== CATALOGUE VALIDATION ==========")
print("Candidate problems:", len(selected))
print("Unique problem indices:", len(np.unique(selected)))
print("Known skill associations:", len(catalogue_skill_indices))
print("Minimum interaction frequency:", int(catalogue_frequency.min()))
print("Maximum interaction frequency:", int(catalogue_frequency.max()))

print("\nFirst 10 candidates:")
print(catalogue_df.head(10).to_string(index=False))

print("\nSaved files:")
print(OUTPUT_NPZ)
print(OUTPUT_CSV)
print("\nCandidate catalogue built and validated successfully.")
