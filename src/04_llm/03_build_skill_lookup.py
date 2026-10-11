"""
03_build_skill_lookup.py

Builds two lookup files from the cleaned interactions:
  1. data/processed/skill_id_to_name.json   -> {skill_id (int): "skill name"}
  2. data/processed/problem_to_skill.json   -> {problem_id (int): {"skill_id": int, "skill_name": str}}

These are used by the LLM lesson planner to convert raw IDs into
human-readable topic names (e.g. Skill 50 -> "Multiplication and Division").
"""

import json
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
INPUT     = ROOT / "data" / "processed" / "clean_interactions.csv"
OUT_SKILL = ROOT / "data" / "processed" / "skill_id_to_name.json"
OUT_PROB  = ROOT / "data" / "processed" / "problem_to_skill.json"

# ── 1. Load only the columns we need (fast on large file) ────────────────────
print("Loading relevant columns from clean_interactions.csv ...")
df = pd.read_csv(INPUT, usecols=["problem_id", "skill_id", "skill"])

print(f"  Total rows loaded : {len(df):,}")
print(f"  Columns           : {df.columns.tolist()}")

# ── 2. Build skill_id -> skill_name map ──────────────────────────────────────
print("\nBuilding skill_id -> skill_name map ...")

skill_df = (
    df[["skill_id", "skill"]]
    .dropna(subset=["skill_id", "skill"])   # keep only rows where both exist
    .drop_duplicates()
)

# There can be slight duplicates where same skill_id maps to slightly different
# strings. Take the most-common name per skill_id.
skill_map_raw = (
    skill_df
    .groupby("skill_id")["skill"]
    .agg(lambda x: x.mode().iloc[0])       # most-common name
    .to_dict()
)

# Convert keys to int for clean JSON
skill_id_to_name = {int(k): str(v) for k, v in skill_map_raw.items()}

print(f"  Unique skill IDs with names : {len(skill_id_to_name)}")

# Show a sample
print("\n  Sample entries (skill_id -> name):")
for sid, name in list(skill_id_to_name.items())[:10]:
    print(f"    {sid:>5}  ->  {name}")

# ── 3. Build problem_id -> {skill_id, skill_name} map ───────────────────────
print("\nBuilding problem_id -> skill map ...")

prob_df = (
    df[["problem_id", "skill_id", "skill"]]
    .dropna(subset=["problem_id", "skill_id", "skill"])
    .drop_duplicates(subset=["problem_id", "skill_id"])
)

# For problems mapped to multiple skills, pick the most common skill
prob_skill = (
    prob_df
    .groupby("problem_id")
    .agg(
        skill_id=("skill_id", lambda x: int(x.mode().iloc[0])),
        skill_name=("skill", lambda x: str(x.mode().iloc[0]))
    )
    .to_dict(orient="index")
)

# Convert outer keys to int
problem_to_skill = {int(k): v for k, v in prob_skill.items()}

print(f"  Unique problem IDs with known skill : {len(problem_to_skill):,}")

# ── 4. Spot-check: look up the exact problem IDs from our MCTS plan ──────────
print("\n--- Spot-check: MCTS recommended problems ---")
MCTS_PLAN = ROOT / "results" / "mcts" / "mcts_fair_plan.csv"
if MCTS_PLAN.exists():
    plan = pd.read_csv(MCTS_PLAN)
    for _, row in plan.iterrows():
        pid  = int(row["original_problem_id"])
        sid  = int(row["original_skill_id"])
        info = problem_to_skill.get(pid)
        name_from_map   = info["skill_name"] if info else "NOT FOUND in lookup"
        name_from_skillid = skill_id_to_name.get(sid, "NOT FOUND in skill_id map")
        print(f"  Step {int(row['step'])}:  problem_id={pid}  skill_id={sid}")
        print(f"           Lookup by problem_id  -> '{name_from_map}'")
        print(f"           Lookup by skill_id    -> '{name_from_skillid}'")
else:
    print("  MCTS plan file not found, skipping spot-check.")

# ── 5. Save to disk ──────────────────────────────────────────────────────────
print("\nSaving lookup files ...")
with open(OUT_SKILL, "w", encoding="utf-8") as f:
    json.dump(skill_id_to_name, f, indent=2, ensure_ascii=False)
print(f"  Saved: {OUT_SKILL.name}")

# problem_to_skill is large; save compactly (no indent)
with open(OUT_PROB, "w", encoding="utf-8") as f:
    json.dump(problem_to_skill, f, ensure_ascii=False)
print(f"  Saved: {OUT_PROB.name}")

print("\nDone! Both lookup files are ready for the LLM lesson planner.")
