import numpy as np
from collections import Counter

base = "data/processed/"
print("Loading categorical inputs...")
data = np.load(base + "categorical_inputs.npz")
problems = data["problem_indices"].reshape(-1)
skills = data["skill_indices"].reshape(-1)
missing = data["skill_missing"].reshape(-1)

valid = problems > 0
problem_counts = np.bincount(problems[valid])

known = valid & (skills > 0) & (missing == 0)

pairs = np.column_stack((problems[known], skills[known]))
unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)

skill_counts = {}
for (problem, skill), count in zip(unique_pairs, pair_counts):
    problem = int(problem)
    skill = int(skill)
    skill_counts.setdefault(problem, Counter())[skill] = int(count)

vocab = np.load(base + "embedding_vocab.npz", allow_pickle=True)
problem_ids = vocab["problem_ids"]

print("\n--- Summary ---")
print("Observed non-padding interactions:", int(valid.sum()))
print("Unique observed problems:", int(np.count_nonzero(problem_counts)))
print("Problems with at least one known skill:", len(skill_counts))
print("Problems with no known skill:", int(np.count_nonzero(problem_counts)) - len(skill_counts))

print("\n--- 15 Most Frequent Problems ---")
print("Index | Original ID | Interactions | Known Skills | Most Common Skill")
top = np.argsort(problem_counts)[::-1]
shown = 0
for idx in top:
    if idx == 0 or idx >= len(problem_counts) or problem_counts[idx] == 0:
        continue

    original_id = str(problem_ids[idx - 1])
    counts = skill_counts.get(int(idx), Counter())

    if counts:
        most_common_skill, skill_count = counts.most_common(1)[0]
        skill_text = str(most_common_skill)
        if len(counts) > 1:
            skill_text += f" (multiple: {len(counts)})"
    else:
        skill_text = "UNKNOWN"

    print(f"{idx} | {original_id} | {problem_counts[idx]} | {len(counts)} | {skill_text}")

    shown += 1
    if shown == 15:
        break
