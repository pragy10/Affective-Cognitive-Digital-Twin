
import numpy as np

BASE = "data/processed/"

print("Loading categorical inputs...")
data = np.load(BASE + "categorical_inputs.npz")

problems = data["problem_indices"].reshape(-1)
skills = data["skill_indices"].reshape(-1)
missing = data["skill_missing"].reshape(-1)

# Count all observed interactions for each problem.
valid = problems > 0
problem_frequency = np.bincount(problems[valid])

# Keep only interactions with a known skill.
known = valid & (skills > 0) & (missing == 0)

known_problem_frequency = np.bincount(
    problems[known],
    minlength=len(problem_frequency)
)

# Find distinct problem-skill combinations.
pairs = np.unique(
    np.column_stack((problems[known], skills[known])),
    axis=0
)

# Number of distinct known skills associated with each problem.
skill_counts = np.bincount(
    pairs[:, 0],
    minlength=len(problem_frequency)
)

# Number of problems associated with multiple known skills.
observed_problems = np.flatnonzero(problem_frequency[1:] > 0) + 1
known_problems = np.flatnonzero(known_problem_frequency[1:] > 0) + 1

multiple_skill_problems = np.sum(skill_counts[known_problems] > 1)

# Load original problem IDs and skill IDs.
vocab = np.load(BASE + "embedding_vocab.npz", allow_pickle=True)
problem_ids = vocab["problem_ids"]
skill_ids = vocab["skill_ids"]

# Convert embedding skill indices to original skill IDs.
skill_index_to_id = {
    int(index): str(skill_id)
    for skill_id, index in zip(
        skill_ids, vocab["skill_indices"]
    )
}

print("\n========== CATALOGUE SUMMARY ==========")
print("Observed problems:", len(observed_problems))
print("Problems with known skills:", len(known_problems))
print("Problems with multiple known skills:", int(multiple_skill_problems))
print(
    "Interactions belonging to known-skill problems:",
    int(known_problem_frequency.sum())
)
print(
    "Share of all interactions covered by known-skill problems:",
    f"{known_problem_frequency.sum() / valid.sum():.2%}"
)

print("\n========== FREQUENT KNOWN-SKILL PROBLEMS ==========")
print("Index | Original ID | Interactions | Skill count | Observed skill IDs")

# Sort by total observed frequency, highest first.
ranked = known_problems[
    np.argsort(problem_frequency[known_problems])[::-1]
]

for index in ranked[:20]:
    # Vocabulary indices are one-based and consecutive.
    original_id = problem_ids[index - 1]

    # Find the distinct known skills for this problem.
    associated_indices = pairs[pairs[:, 0] == index, 1]
    associated_ids = [
        skill_index_to_id[int(skill)]
        for skill in associated_indices
    ]

    print(
        f"{index} | {original_id} | "
        f"{problem_frequency[index]} | "
        f"{len(associated_indices)} | "
        f"{associated_ids}"
    )

print("\nCatalogue analysis completed.")
